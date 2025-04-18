import os
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import torch


# 配置参数
class Config:
    MODEL_PATH = "/data3/dxw/data/sft_role_megred/"
    EMBEDDING_MODEL_PATH = "/data3/dxw/model/text2vec-large-chinese"
    DATA_PATH = "/data3/dxw/role/sidamingzhu.txt"

    # 设备配置
    LLM_DEVICE = "cuda:2"
    EMBEDDING_DEVICE = "cuda:1"

    # 文本处理参数
    CHUNK_SIZE = 50
    CHUNK_OVERLAP = 20
    MAX_DOC_LENGTH = 2000

    # 生成参数
    GENERATION_CONFIG = {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "num_return_sequences": 1
    }


# 自定义提示模板
CUSTOM_PROMPT_TEMPLATE = """基于以下上下文回答问题。只需要给出最终答案，不要任何解释。

上下文：
{context}

问题：{question}
简洁的答案："""


def setup_environment():
    """配置调试环境"""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    torch.cuda.empty_cache()


def process_documents() -> List[Document]:
    """文档加载与处理"""
    loader = TextLoader(Config.DATA_PATH, encoding="utf-8")
    raw_docs = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=lambda x: len(x),
        separator="\n"
    )
    return text_splitter.split_documents([
        Document(page_content=doc.page_content[:Config.MAX_DOC_LENGTH])
        for doc in raw_docs
    ])


def initialize_generation_pipeline() -> HuggingFacePipeline:
    """加载生成模型到cuda:2"""
    tokenizer = AutoTokenizer.from_pretrained(
        Config.MODEL_PATH,
        padding_side="left",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    text_generation_pipeline = pipeline(
        task="text-generation",
        model=Config.MODEL_PATH,
        tokenizer=tokenizer,
        device=2,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "pad_token_id": tokenizer.eos_token_id,
        },
    ** Config.GENERATION_CONFIG
    )
    print(f"LLM设备: {text_generation_pipeline.device}")
    return HuggingFacePipeline(pipeline=text_generation_pipeline)


def create_vector_store(docs: List[Document]) -> Chroma:
    """创建向量数据库到cuda:1"""
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_PATH,
        model_kwargs={"device": Config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma.from_documents(docs, embeddings)


def build_qa_system(llm: HuggingFacePipeline, retriever) -> RetrievalQA:
    """构建问答系统"""
    custom_prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt},
        verbose=False  # 关闭详细日志
    )


def main():
    """主执行流程"""
    setup_environment()
    print("\n" + "=" * 50)
    print(f"LLM设备: {Config.LLM_DEVICE}")
    print(f"Embedding设备: {Config.EMBEDDING_DEVICE}")
    print("=" * 50 + "\n")

    try:
        # 初始化组件
        llm = initialize_generation_pipeline()
        docs = process_documents()
        db = create_vector_store(docs)
        retriever = db.as_retriever()
        qa_system = build_qa_system(llm, retriever)

        query = "白骨精被打死几次？"

        # 检索文档
        retrieved_docs = retriever.get_relevant_documents(query)

        # 打印问题和检索结果
        print("问题:", query)
        print("\n检索到的文档内容:")
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"[文档{i}] {doc.page_content}")

        # 获取并打印回答
        result = qa_system.invoke({"query": query})
        print("\nLLM的回答:", result["result"].strip())

    except Exception as e:
        print(f"系统错误: {str(e)}")


if __name__ == "__main__":
    main()