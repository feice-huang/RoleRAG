from typing import List, Optional, Union
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import json
import os
import pickle

class Retriever:
    def __init__(self, config):
        """
        初始化 Retrieve 系统
        :param config: 配置对象，包含文件路径、向量保存路径等
        """
        self.config = config
        self.vector_store: Optional[FAISS] = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators=["\n\n"])
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.get("embedding_model", "/data/hfc/checkpoints/text2vec-large-chinese"),
            encode_kwargs={"normalize_embeddings": True}
        )
        self.all_docs: List[Document] = []  # 保存原始切分文档（用于上下文追溯）
        

    def load_files(self, file_paths: List[str]) -> List[Document]:
        """根据给定的文件路径列表加载和处理文档"""
        docs = []
        for file_path in file_paths:
            if file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                # 按照 \n\n（两个换行符）分割段落
                paragraphs = text.split("\n\n")
                for paragraph in paragraphs:
                    docs.append(Document(page_content=paragraph.strip(), metadata={"source": file_path}))
            elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.endswith(".jsonl"):
                        lines = f.readlines()
                        for line in lines:
                            data = json.loads(line.strip())
                            text = json.dumps(data, ensure_ascii=False)
                            splits = self.text_splitter.split_text(text)
                            for chunk in splits:
                                docs.append(Document(page_content=chunk, metadata={"source": file_path}))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                text = json.dumps(item, ensure_ascii=False)
                                splits = self.text_splitter.split_text(text)
                                for chunk in splits:
                                    docs.append(Document(page_content=chunk, metadata={"source": file_path}))
                        else:
                            text = json.dumps(data, ensure_ascii=False)
                            splits = self.text_splitter.split_text(text)
                            for chunk in splits:
                                docs.append(Document(page_content=chunk, metadata={"source": file_path}))
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        return docs

    def create_vector_store(self, docs: List[Document]):
        """创建向量数据库"""
        self.vector_store = FAISS.from_documents(docs, self.embedding_model)
        self.all_docs = docs  # 保存切分后的原始数据，用于追溯上下文

    def save_vector_store(self, save_dir: str):
        """保存向量数据库和 all_docs 到磁盘"""
        if self.vector_store is None:
            raise ValueError("No vector store to save.")
        self.vector_store.save_local(save_dir)
        # 保存 all_docs
        with open(os.path.join(save_dir, "all_docs.pkl"), "wb") as f:
            pickle.dump(self.all_docs, f)

    def load_vector_store(self, load_dir: str, allow_dangerous_deserialization: bool = False):
        """
        从磁盘加载向量数据库和 all_docs
        
        Args:
            load_dir: 加载目录路径
            allow_dangerous_deserialization: 是否允许危险的反序列化操作
        """
        self.vector_store = FAISS.load_local(
            load_dir, 
            self.embedding_model,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        
        # 加载 all_docs
        pkl_path = os.path.join(load_dir, "all_docs.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                if allow_dangerous_deserialization:
                    self.all_docs = pickle.load(f)
                else:
                    raise ValueError(
                        "加载 all_docs.pkl 需要反序列化，这可能存在安全风险。"
                        "如果您信任此文件来源，请设置 allow_dangerous_deserialization=True"
                    )
        else:
            raise FileNotFoundError(f"No saved document context found at {pkl_path}")

    def retrieve_top_k(self, query: str, k: int, with_context: bool = False) -> List[Document]:
        """检索与查询相关的 top-k 文档，并可附加上下文"""
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Please run create_vector_store() or load_vector_store() first.")
        
        retrieved_docs = self.vector_store.similarity_search(query, k=k)

        if not with_context:
            return retrieved_docs
        
        docs_with_context = []
        for doc in retrieved_docs:
            doc.metadata["original_content"] = doc.page_content
            context_doc = self._add_context(doc)
            docs_with_context.append(context_doc)
        return docs_with_context

    def _add_context(self, doc: Document, window_size: int = 3) -> Document:
        """给检索到的文档添加上下文"""
        if not self.all_docs:
            return doc
        
        # 用内容来模糊查找位置，而不是对象比较
        target_content = doc.page_content.strip()
        idx = -1
        for i, candidate_doc in enumerate(self.all_docs):
            if target_content in candidate_doc.page_content:
                idx = i
                break
        
        if idx == -1:
            # 如果找不到，直接返回原文
            return doc
        
        start = max(idx - window_size, 0)
        end = min(idx + window_size + 1, len(self.all_docs))
        surrounding_texts = [self.all_docs[i].page_content for i in range(start, end)]
        combined_text = "\n".join(surrounding_texts)

        return Document(page_content=combined_text, metadata=doc.metadata)

