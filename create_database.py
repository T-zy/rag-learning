# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
# import openai 
from dotenv import load_dotenv
import os
import shutil
from langchain.document_loaders import PyPDFLoader
# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
# openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"


def main():
    # 直接加载和分块文档
    chunks = load_and_split_document()
    save_to_chroma(chunks)


def load_and_split_document():
    # 加载PDF文档（原项目逻辑）
    loader = PyPDFLoader("/root/langchain-rag-tutorial/data/books/1718342011301056183.pdf")
    documents = loader.load()
    
    # ------------------- 新增：医疗领域文档分块策略 -------------------
    # 按医学章节分块（如每个章节作为一个块），并添加元数据标签
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # 适合医学文档的块大小
        chunk_overlap=200,
        separators=["\n### ", "\n#### ", "\n2.", "\n3.", "\n- ", ". ", "?", "!"]
    )
    
    # 自定义医学章节识别函数（根据共识文档结构）
    def get_medical_metadata(page_content):
        # 示例：根据标题判断章节类型
        if "诊断标准" in page_content or "表2" in page_content:
            return {"section": "diagnosis", "topic": "诊断标准"}
        elif "治疗目标" in page_content or "表3" in page_content:
            return {"section": "treatment", "topic": "治疗目标"}
        elif "急症" in page_content or "并发症" in page_content:
            return {"section": "emergency", "topic": "急症处理"}
        else:
            return {"section": "general", "topic": "综合管理"}
    
    # 分块并添加元数据
    split_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            metadata = get_medical_metadata(chunk)
            split_docs.append(
                Document(page_content=chunk, metadata={**doc.metadata, **metadata})
            )
    
    return split_docs

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, SentenceTransformerEmbeddings(model_name="/root/.cache/modelscope/hub/models/sungw111/text2vec-base-chinese-sentence"), persist_directory=CHROMA_PATH
    )#这里模型做了更改，原项目是all-MiniLM-L6-v2
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
