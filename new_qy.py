from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np
from langchain_chroma import Chroma
from langchain.schema import BaseRetriever, Document
from pydantic import PrivateAttr

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
from langchain.llms import HuggingFacePipeline

load_dotenv()

# ------------------- System Prompt -------------------
MEDICAL_SYSTEM_PROMPT = """
你是一位专注于高血压、2型糖尿病和血脂异常（"三高"）防治的医疗知识顾问。你的回答需严格基于《成人高血压合并2型糖尿病和血脂异常基层防治中国专家共识（2024年版）》，确保所有信息准确引用共识中的推荐意见（如"2.7推荐意见"）或数据（如"表3控制目标"）。

**回答原则：**
1. **医学严谨性**：优先使用共识中的诊断标准、治疗目标和干预措施（如血压控制目标<130/80 mmHg），避免模糊表述。
2. **结构化输出**：采用分级标题（###）、列表（- 或数字编号）和表格（如有多维度对比），关键数据加粗（如"BMI控制在18.5~24.0 kg/m²"）。
3. **风险提示**：若问题涉及急症（如低血糖、高血压危象），需明确标注"紧急转诊指征"并给出初步处理建议。
4. **伦理边界**：对于超出基层管理范围的问题，需提示"建议咨询专科医生"。

**知识源引用规范：**
- 共识章节引用格式："根据共识2.3节预防措施..."
- 表格数据引用格式："参考表3'三高'控制目标..."
- 推荐意见等级标注："(1A类推荐)""(2B类推荐)"
"""

# ------------------- Prompt 模板 -------------------
DIAGNOSIS_PROMPT_TEMPLATE = """
你是一位医疗助手，专注于高血压、2型糖尿病和血脂异常（“三高”）的基层防治，所有回答基于《中国专家共识（2024年）》。

### 检索到的知识片段
{context}

### 问题分析
用户询问{disease}的诊断标准，请根据共识第2.4节及表2内容，分指标列出诊断标准，需包含检测值、检测方法及注意事项。

### 回答要求
- 按"指标类型→检测标准→推荐等级"结构化输出
- 每个标准标注推荐等级（如"1B类推荐"）
- 注明数值单位（如mmHg、mmol/L）和换算关系（1 mmHg=0.133 kPa）

参考格式：
- **指标类型：高血压**
  - 检测标准：SBP≥140 mmHg和/或DBP≥90 mmHg（诊室3次测量）
  - 检测方法：诊室血压、家庭血压、动态血压监测
  - 推荐等级：1B类推荐

### 原始问题
{question}
"""


TREATMENT_PROMPT_TEMPLATE = """
### 问题分析
用户需要{condition}的治疗建议，需结合共识{section}内容，重点回答：
1. 一线治疗措施及优先选用依据（标注推荐等级）
2. 联合治疗指征（如指标阈值时的调整）
3. 特殊人群注意事项
4. 潜在风险提示（如药物相互作用）

### 回答要求
- 用"治疗分类→适用场景→注意事项"三级结构
- 关键数据加粗显示
- 顶部添加风险提示（如涉及急症处理需标注"紧急转诊"）

### 原始问题
{question}
"""

# ------------------- PromptTemplate 构建 -------------------
DIAGNOSIS_PROMPT = PromptTemplate(
    input_variables=["disease", "question", "context"],
    template=DIAGNOSIS_PROMPT_TEMPLATE
)

TREATMENT_PROMPT = PromptTemplate(
    input_variables=["condition", "section", "question"],
    template=MEDICAL_SYSTEM_PROMPT + TREATMENT_PROMPT_TEMPLATE
)

DEFAULT_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=MEDICAL_SYSTEM_PROMPT + "\n### 问题：{question}\n### 请基于共识规范回答："
)

# ------------------- prompt选择器 -------------------
def prompt_selector(question):
    if "诊断标准" in question or "如何诊断" in question:
        return DIAGNOSIS_PROMPT
    elif "治疗" in question or "用药" in question or "干预" in question:
        return TREATMENT_PROMPT
    else:
        return DEFAULT_PROMPT

# ------------------- 混合检索逻辑 -------------------
def hybrid_retrieve_and_rerank(query, vectordb, embedder, reranker, top_k=3, bm25_k=5, vector_k=5):
    all_docs = vectordb.get()['documents']
    if not all_docs:
        raise ValueError("向量数据库为空")

    tokenized_corpus = [doc.split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_results = bm25.get_top_n(query.split(), all_docs, n=bm25_k)

    query_vec = embedder.encode(query)
    doc_vecs = embedder.encode(all_docs)
    sims = np.dot(doc_vecs, query_vec) / (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(query_vec))
    topk_idx = np.argsort(sims)[-vector_k:][::-1]
    vector_results = [all_docs[i] for i in topk_idx]

    candidates = list(dict.fromkeys(bm25_results + vector_results))
    rerank_inputs = [[query, doc] for doc in candidates]
    scores = reranker.predict(rerank_inputs)
    reranked = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]
    return reranked[:top_k]

# ------------------- QA链构建 -------------------
def get_medical_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="/root/bge-large-zh")
    model1 = SentenceTransformer("/root/bge-large-zh")
    vectordb = Chroma(
        persist_directory="./chroma",
        embedding_function=embeddings
    )

    reranker = CrossEncoder('/root/.cache/modelscope/hub/models/BAAI/bge-reranker-base')

    model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    def custom_retriever(query):
        top_chunks = hybrid_retrieve_and_rerank(query, vectordb, model1, reranker, top_k=3)
        return [Document(page_content=chunk) for chunk in top_chunks]

    class CustomRetriever(BaseRetriever):
        _retriever_func = PrivateAttr()

        def __init__(self, retriever_func):
            super().__init__()
            self._retriever_func = retriever_func

        def _get_relevant_documents(self, query, *, run_manager=None, **kwargs):
            return self._retriever_func(query)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=CustomRetriever(custom_retriever),
        return_source_documents=False
    )
    return qa_chain

# ------------------- 主函数 -------------------
if __name__ == "__main__":
    question = "三高的诊断标准是什么？"
    prompt = prompt_selector(question)

    # 假设这是你检索到的上下文内容
    context_text = "根据共识2.4节，高血压的诊断标准为..."

    # 定义所有变量
    values = {
        "disease": "三高",
        "condition": "三高",
        "section": "2.4",
        "question": question,
        "context": context_text
    }

    # 格式化最终 prompt
    final_prompt = prompt.format(**{k: values[k] for k in prompt.input_variables})

    qa_chain = get_medical_qa_chain()
    result = qa_chain({"query": final_prompt})

    print("\n===== 医疗RAG回答 =====")
    print(result["result"])
