import os
import re
from typing import List
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
import torch


class Config:
    MODEL_PATH = "/data/hfc/checkpoints/Llama-3.1-8B-Instruct"  # SFT模型路径
    EMBEDDING_MODEL_PATH = "/data/hfc/checkpoints/text2vec-large-chinese"  # 嵌入模型路径
    DATA_PATH = "/data/hfc/mydata/wiki/wiki_刘星.txt"  # 数据文件路径

    LLM_DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
    EMBEDDING_DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"

    CHUNK_SIZE = 30
    CHUNK_OVERLAP = 10

    GENERATION_CONFIG = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True
    }


class LiuXingAgent:
    def __init__(self):
        self.setup_environment()
        self.llm = self.initialize_llm()
        self.qa_system = self.build_rag_system()

    def debug_print(self, message: str):
        """调试信息输出"""
        print(f"[DEBUG] {message}")

    def setup_environment(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.cuda.empty_cache()

    def initialize_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_PATH,
            padding_side="left",
            use_fast=False
        )
        tokenizer.pad_token = tokenizer.eos_token

        text_pipeline = pipeline(
            task="text-generation",
            model=Config.MODEL_PATH,
            tokenizer=tokenizer,
            device=Config.LLM_DEVICE,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            **Config.GENERATION_CONFIG
        )
        return HuggingFacePipeline(pipeline=text_pipeline)

    def build_rag_system(self):
        self.debug_print(f"正在加载知识库文件：{Config.DATA_PATH}")
        loader = TextLoader(Config.DATA_PATH, encoding="utf-8")
        documents = loader.load()
        self.debug_print(f"原始文档长度：{len(documents)}篇")

        text_splitter = CharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separator="\n\n"
        )
        docs = text_splitter.split_documents(documents)
        self.debug_print(f"分割后文档数量：{len(docs)}块")

        embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_PATH,
            model_kwargs={"device": Config.EMBEDDING_DEVICE},
        )
        db = Chroma.from_documents(docs, embeddings)
        self.debug_print("向量数据库构建完成")

        prompt_template = """基于以下背景信息：
{context}

回答问题：{question}
请直接生成简短事实性答案（使用列表格式）："""

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
            },
            return_source_documents=False
        )

    def pre_retrieve(self, question: str) -> str:
        """问题预检索，激活SFT模型进行问题简短浓缩"""
        prompt = f"请你将原始问题，简短浓缩为主谓宾状语的短句：{question}"

        self.debug_print("\n=== 预检索阶段 ===")
        self.debug_print(f"原始问题：{question}")
        self.debug_print(f"生成提示模板：\n{prompt}")

        response = self.llm(prompt)  # 使用 __call__ 方法，而不是 run

        # 清理输出，只保留生成的结果，不含输入的prompt部分
        self.debug_print(f"模型输出类型：{type(response)}")
        if isinstance(response, list) and len(response) > 0:
            generated = response[0].get('generated_text', '')
        else:
            generated = str(response)

        self.debug_print(f"模型原始输出：\n{generated}")

        # 清理输出：去除输入提示，保留简短浓缩后的问题
        cleaned_output = re.sub(r'请你将原始问题，简短浓缩为主谓宾状语的短句：', '', generated, flags=re.DOTALL)
        self.debug_print(f"清理后的输出：\n{cleaned_output}")

        return cleaned_output.strip()

    def retrieve_answers(self, query: str) -> str:
        """检索回答阶段，基于简短浓缩后的问题进行RAG检索"""
        self.debug_print("\n=== 检索回答阶段 ===")
        result = self.qa_system({"query": query})
        return result.get('result', '').strip() if result else ""

    def final_generation(self, rag_ans: str) -> str:
        """风格转换生成阶段"""
        self.debug_print("\n=== 风格转换阶段 ===")
        prompt = f"你是刘星，需用他的语气风格改写输入句子：{rag_ans}"

        self.debug_print(f"转换提示模板：\n{prompt}")

        try:
            response = self.llm(prompt)  # 使用 __call__ 方法，而不是 run

            self.debug_print(f"模型输出类型：{type(response)}")
            if isinstance(response, list) and len(response) > 0:
                output = response[0].get('generated_text', '')
            else:
                output = str(response)

            self.debug_print(f"模型原始输出：\n{output}")

            final_answer = output.split("刘星风格回答：")[-1].split("\n")[0].strip()
            final_answer = re.sub(r'[$$$$【】「」]', '', final_answer)

            if not final_answer.endswith(('？', '！')):
                final_answer += "，您说是不是啊？"

            self.debug_print(f"最终处理后答案：{final_answer}")
            return final_answer

        except Exception as e:
            self.debug_print(f"生成失败：{str(e)}")
            return "真得容我仔细想想！"

    def process_query(self, user_question: str) -> str:
        try:
            self.debug_print("\n" + "=" * 50)
            self.debug_print(f"开始处理问题：{user_question}")

            # 第一阶段：问题预检索
            simplified_question = self.pre_retrieve(user_question)
            if not simplified_question:
                self.debug_print("问题浓缩失败")
                return "这事儿我得再琢磨琢磨..."

            # 第二阶段：RAG检索
            rag_ans = self.retrieve_answers(simplified_question)
            if "暂缺" in rag_ans:
                self.debug_print("关键信息缺失")

            # 第三阶段：风格转换
            final_answer = self.final_generation(rag_ans)
            return final_answer

        except Exception as e:
            self.debug_print(f"全局异常：{str(e)}")
            return "哎呦喂，这事儿整的有点岔劈了！"


if __name__ == "__main__":
    agent = LiuXingAgent()

    test_questions = [
        "小雨：哥，在家里的客厅里你为什么会突然想要一个女朋友，并且自言自语说出来呢？",
        "小雪：刘星，你和夏雨是什么时候发现我有男朋友的？",
        "小雨：哥，刚才爸爸妈妈说小雪开始变得乖巧了，你觉得是不是因为他们的宽容和理解起了作用啊？"
    ]
    for idx, q in enumerate(test_questions, 1):
        print(f"\n{'=' * 30} 测试用例 {idx} {'=' * 30}")
        print(f"用户提问：{q}")
        answer = agent.process_query(q)
        print(f"\n刘星回答：{answer}")
        print("=" * 50)
