# Copyright 2025 THUDM and the LlamaFactory team.
#
# This code is inspired by the THUDM's ChatGLM implementation.
# https://github.com/THUDM/ChatGLM-6B/blob/main/cli_demo.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# modified feice Apr 23, 2025 at 14:12
将run_cot的第一步recall更换成网络搜索（Google Custom Search API），在函数内部设置代理以防止被墙。

# modified feice Apr 27, 2025 at 10:51
加入本地知识库的retrieve功能

# modified feice Apr 28, 2025 at 10:34
实现大规模处理

"""
import asyncio
import os
import json
from collections.abc import AsyncGenerator, Generator
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

from ..extras.constants import EngineName
from ..extras.misc import torch_gc
from ..hparams import get_infer_args
from .hf_engine import HuggingfaceEngine
from .sglang_engine import SGLangEngine
from .vllm_engine import VllmEngine
from ..retrieve import Retriever


if TYPE_CHECKING:
    from ..data.mm_plugin import AudioInput, ImageInput, VideoInput
    from .base_engine import BaseEngine, Response


def _start_background_loop(loop: "asyncio.AbstractEventLoop") -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    r"""General class for chat models. Backed by huggingface or vllm engines.

    Supports both sync and async methods.
    Sync methods: chat(), stream_chat() and get_scores().
    Async methods: achat(), astream_chat() and aget_scores().
    """

    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        if model_args.infer_backend == EngineName.HF:
            self.engine: BaseEngine = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        elif model_args.infer_backend == EngineName.VLLM:
            self.engine: BaseEngine = VllmEngine(model_args, data_args, finetuning_args, generating_args)
        elif model_args.infer_backend == EngineName.SGLANG:
            self.engine: BaseEngine = SGLangEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError(f"Unknown backend: {model_args.infer_backend}")

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        r"""Get a list of responses of the chat model."""
        task = asyncio.run_coroutine_threadsafe(
            self.achat(messages, system, tools, images, videos, audios, **input_kwargs), self._loop
        )
        return task.result()

    async def achat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> list["Response"]:
        r"""Asynchronously get a list of responses of the chat model."""
        return await self.engine.chat(messages, system, tools, images, videos, audios, **input_kwargs)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        r"""Get the response token-by-token of the chat model."""
        generator = self.astream_chat(messages, system, tools, images, videos, audios, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        images: Optional[list["ImageInput"]] = None,
        videos: Optional[list["VideoInput"]] = None,
        audios: Optional[list["AudioInput"]] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        r"""Asynchronously get the response token-by-token of the chat model."""
        async for new_token in self.engine.stream_chat(
            messages, system, tools, images, videos, audios, **input_kwargs
        ):
            yield new_token

    def get_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        r"""Get a list of scores of the reward model."""
        task = asyncio.run_coroutine_threadsafe(self.aget_scores(batch_input, **input_kwargs), self._loop)
        return task.result()

    async def aget_scores(
        self,
        batch_input: list[str],
        **input_kwargs,
    ) -> list[float]:
        r"""Asynchronously get a list of scores of the reward model."""
        return await self.engine.get_scores(batch_input, **input_kwargs)


def run_chat() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    chat_model = ChatModel()
    messages = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            torch_gc()
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})

# modified feice Apr 23, 2025 at 14:11
import requests
import time

GOOGLE_API_URL = "https://www.googleapis.com/customsearch/v1"
GOOGLE_API_KEY = "AIzaSyDO3hlUuK9CugxCIQf86czKk-_K_NeHvuA"  # 替换为你的 Google API 密钥
GOOGLE_CX = "87b42182cc13c4b23"  # 替换为你的 Google Custom Search Engine ID

def google_search(query, num_results=4, timeout=5):
    """
    使用 Google Custom Search JSON API 执行查询并返回结果。

    :param query: 搜索的关键词或主题
    :param num_results: 返回的搜索结果数量（默认 4）
    :param timeout: 请求超时时间（默认 5 秒）
    :return: 检索到的文本和搜索耗时
    """
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "num": num_results
    }

    # 设置代理
    proxies = {
        "http": "http://219.223.184.164:7890",
        "https": "http://219.223.184.164:7890"
    }

    try:
        start_time = time.time()
        response = requests.get(GOOGLE_API_URL, params=params, timeout=timeout, proxies=proxies)
        response.raise_for_status()  # 如果响应状态码不是 200，抛出异常
        search_time = time.time() - start_time

        # 解析响应 JSON
        response_json = response.json()
        items = response_json.get("items", [])
        
        # 提取搜索结果
        results = []
        for item in items:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            results.append(f"{title}: {snippet}")
        
        # 将结果合并为单个字符串
        result_text = " ".join(results)
        return result_text, search_time

    except requests.exceptions.RequestException as e:
        print(f"Error during Google search: {e}")
        return "", 0

# 定义每个任务的system以及instruction prompt
def get_recall_prompt(question: str) -> str:
    return f"""下面是一段关于家有儿女和刘星的问题，这段问题将“你”视为“刘星”。
请为我将问题转化为适用于网络搜索的陈述，将所有代词替换为实体。
严格遵循示例中的格式，不需要多余分析，避免诸如\"以下是答案：\"之类的陈述。\n

示例输入：你喜欢你妈妈吗？
示例输出格式：刘星喜欢刘梅吗？

需要转化的问题：
{question}
"""

def get_cot_prompt(question: str, observation: str) -> str:
    return f"""你正在扮演刘星，请以刘星的身份回答问题，注意不要虚构事实，如果是不知道的事情，需要表现出疑惑。
问题：
{question}
可能的参考信息：
{observation}
"""

def get_style_prompt(role, question: str) -> str:
    return f"""你正在扮演 {role}，你需要将下面的句子转写成 {role} 的口吻
{question}
"""

# modified feice Apr 19, 2025 at 20:04
def run_cot() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    # 初始化三个模型
    print("正在加载模型...")
    rewrite_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/GLM-4-9B-0414",
        "template": "glm4"
    })
    cot_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/GLM-4-9B-0414",
        "adapter_name_or_path": "/data/hfc/RoleRAG/saves/刘星_glm4_cot_800/glm4_8b_sft_lora/TorchTrainer_a7d31_00000_0_2025-04-25_10-54-50/checkpoint_000000/checkpoint",
        "template": "glm4"
    })
    style_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/GLM-4-9B-0414",
        "adapter_name_or_path": "/data/hfc/RoleRAG/saves/刘星_glm4_style/glm4_8b_sft_lora/TorchTrainer_436ee_00000_0_2025-04-29_10-32-58/checkpoint_000001/checkpoint",
        "template": "glm4"
    })
    print("模型加载完成！")

    # 初始化
    retriever = Retriever(config={
        "embedding_model": "/data/hfc/checkpoints/text2vec-large-chinese"
    })

    role = "刘星"
    world = "家有儿女"

    file_paths = [
        f"/data/hfc/RoleRAG/mydata/input/wiki/wiki_{role}.txt",
        f"/data/hfc/RoleRAG/mydata/input/wiki/wiki_{world}.txt"
        # "/data/hfc/datasets/RoleAgentBench/家有儿女 S1E1/profiles/刘星.jsonl"
    ]

    docs = retriever.load_files(file_paths)
    retriever.create_vector_store(docs)

    # 保存
    retriever.save_vector_store(f"/data/hfc/faiss_store/{world}_{role}")

    # 定义问题列表
    questions = [
        "你如何看待人工智能？",
        "你为什么喜欢捣蛋？",
        "你喜欢玩原神吗？",
        "你知道刘梅的生日吗？",
        "你喜欢刘梅吗？",
        "你喜欢刘星吗？",
        "你还记得加利福尼亚的大蜘蛛吗？"
    ]

    for question in questions:
        print(f"\nUser: {question}")

        # 阶段 1：Recall 模型
        print("recall instruction: ", get_recall_prompt(question))
        recall_messages = [{"role": "user", "content": get_recall_prompt(question)}]
        print("Rewrite Assistant: ", end="", flush=True)
        observation = ""
        for new_text in rewrite_model.stream_chat(recall_messages, system="你是一个人工智能问题重写助手，按照规则重写问题"):
            print(new_text, end="", flush=True)
            observation += new_text
        print("\n"+"="*20)
        torch_gc()  # 清理显存
        retrieved_docs = retriever.retrieve_top_k(question, k=2, with_context=False)

        for i, doc in enumerate(retrieved_docs, 1):
            print(f"=== 第 {i} 条检索结果 ===")
            print(f"元数据: {doc.metadata}")
            print()
            # 合并检索到的文本
            observation += "\n" + doc.page_content

        # 阶段 2：CoT 模型
        print("cot instruction: ", get_cot_prompt(question, observation))
        cot_messages = [{"role": "user", "content": get_cot_prompt(question, observation)}]
        print("CoT Assistant: ", end="", flush=True)
        cot_response = ""
        for new_text in cot_model.stream_chat(cot_messages, system="你是一个角色扮演专家，请以【问题重述】【实体确认】【逻辑推理】【分析回答】【最终回答】的顺序，以扮演角色的身份回答问题"):
            print(new_text, end="", flush=True)
            cot_response += new_text
        print("\n"+"="*20)
        torch_gc()  # 清理显存

        # 阶段 3：Style 模型
        cot_last_response = cot_response.split("\n")[-1].strip()
        print("style instruction: ", get_style_prompt("刘星", cot_last_response))
        style_messages = [{"role": "user", "content": get_style_prompt("刘星", cot_last_response)}]
        # style_messages = [{"role": "user", "content": style_prefix + cot_response}]
        print("Style Assistant: ", end="", flush=True)
        style_response = ""
        for new_text in style_model.stream_chat(style_messages, system="你是一个语言改写助手，将这段语句转换为扮演人物的说话语气。"):
            print(new_text, end="", flush=True)
            style_response += new_text
        print("\n"+"="*20)
        torch_gc()  # 清理显存

        print("Final Output: ", style_response)
        print("History has been removed.\n")


# 定义每个任务的system以及instruction prompt
def get_rag_prompt(role: str, profile: str, question: str, observation: str) -> str:
    """
    生成用于 RAG（检索增强生成）的提示（prompt）
    :param observation: 当前的上下文
    :param question: 用户提问的问题
    :return: 返回生成的提示文本
    """
    return f"""据下面的角色扮演信息，判断信息是否足够回答问题，如果信息不足，需要给出用于解决问题的辅助问题，辅助问题需要非常简单，专注于关键信息。
如果信息足以回答，则输出:
[Answer] <回答内容>
如果信息不足以回答，则输出：
[Retrieve] <辅助问题> 
你应该更多地认为问题能够被解答，除非确实缺少关键信息。注意只需要给出答案，不需要任何解释和分析。
你需要扮演的角色是：
{role}
你需要扮演的角色的信息：
{profile}
你需要回答的问题是：
{question}
可能的参考信息：
{observation}
"""


def get_final_answer_prompt(role: str, profile: str, question: str, observation: str) -> str:
    return f"""根据下面的角色扮演信息，回答问题。注意只需要给出答案，不需要任何解释和分析。
你需要扮演的角色是：
{role}
你需要扮演的角色的信息：
{profile}
你需要回答的问题是：
{question}
可能的参考信息：
{observation}

输出格式：
[Answer] <回答内容>
"""


def get_style_prompt(role: str, sentence: str) -> str:
    return f"""你需要将下面的句子转写成对应角色的口吻，注意只需要给出答案，不需要任何解释和分析。
你需要扮演的角色是：
{role}
你需要转写的句子是：
{sentence}

输出格式：
[Style] <回答内容>
"""
    
# modified feice Apr 24, 2025 at 21:24
def run_rag() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    # 设置角色与世界
    role = "刘星"
    world = "家有儿女"
    profile = "刘星，刘梅之子。初中生（在第四部升上高中生），成绩（尤其化学）常令刘梅头痛。身材看似“瘦弱”，体育倒很不错。爱好广泛但大多都只折腾一时。一家的活宝，大多数麻烦的制造者。为人仗义，脑子里经常有些新奇的想法，里面有好主意也有馊主意。"

    # 初始化模型
    print("正在加载模型...")
    rag_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/GLM-4-9B-0414",
        "template": "glm4"
    })
    
    print("模型加载完成！")

    # 初始化检索器
    retriever = Retriever(config={
        "embedding_model": "/data/hfc/checkpoints/text2vec-large-chinese"
    })

    # 加载文件路径
    file_paths = [
        "/data/hfc/RoleRAG/mydata/input/wiki/wiki_刘星.txt",
        "/data/hfc/RoleRAG/mydata/input/wiki/wiki_家有儿女.txt",
        "/data/hfc/datasets/RoleAgentBench/家有儿女 S1E1/profiles/刘星.jsonl"
    ]

    # 加载文件并创建向量数据库
    docs = retriever.load_files(file_paths)
    retriever.create_vector_store(docs)

    # 保存向量数据库
    retriever.save_vector_store(f"/data/hfc/faiss_store/{world}_{role}_rag")

    # 定义问题列表
    questions = [
        "你如何看待人工智能？",
        "你为什么喜欢捣蛋？",
        "你喜欢玩原神吗？",
        "你知道刘梅的生日吗？",
        "你喜欢刘梅吗？",
        "你喜欢刘星吗？",
        "你还记得加利福尼亚的大蜘蛛吗？"
    ]

    # 初始化观察状态为空
    observation = ""

    # 循环处理每个问题
    for question in questions:
        print(f"\n问题：{question}")
        
        # 生成 RAG 提示并传递给模型
        messages = [{"role": "user", "content": get_rag_prompt(role, profile, question, observation)}]
        model_response = ""
        
        max_attempts = 5  # 最大尝试次数
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            # 使用流式输出调用模型
            for new_response in rag_model.stream_chat(messages):
                print(new_response, end="", flush=True)
                model_response += new_response
            print("\n" + "*"*20)
            print(model_response)
            
            # 去除开头的"\n"和" "
            model_response = model_response.lstrip("\n").lstrip(" ")

            if model_response.startswith("[Retrieve]"):
                # 调用检索器进行检索
                print("开始检索...")
                retrieved_docs = retriever.retrieve_top_k(question, k=3, with_context=False)
                
                # 将检索到的文档加入观察状态
                retrieval_text = "\n".join([doc.page_content for doc in retrieved_docs])
                observation += retrieval_text  # 更新观察状态
                
                print("检索结果已加入观察状态.")
                print(f"检索结果：\n{retrieval_text}")
            elif model_response.startswith("[Answer]"):
                # 如果模型返回了答案，跳出循环
                print("模型返回答案，停止进一步请求。")
                break  # 跳出循环
            else:
                print("无效的模型响应，无法处理。")
        
        if model_response.startswith("[Answer]"):
            # 解析模型的回答
            answer = model_response[len("[Answer]"):].strip()
            print(f"模型回答：{answer}")

            # 通过风格调整 prompt 进行风格化
            style_prompt = get_style_prompt(role, answer)
            style_response = ""
            for new_response in rag_model.stream_chat({"role": "user", "content": style_prompt}):
                print(new_response, end="", flush=True)
                style_response += new_response
            
            style_answer = style_response[len("[Style]"):].strip()

            print(f"风格化回答：{style_answer}")
            observation = ""  # 清空观察状态
            torch_gc()  # 清理显存
        else:
            # 如果达到最大尝试次数且没有返回有效答案，调用最终回答提示
            print(f"模型未能生成有效的答案，调用最终回答生成。")
            final_answer_prompt = get_final_answer_prompt(role, profile, question, observation)
            final_answer = ""
            for new_response in rag_model.stream_chat([{"role": "user", "content": final_answer_prompt}]):
                print(new_response, end="", flush=True)
                final_answer += new_response
            print(f"\n最终回答：{final_answer}")
            observation = ""  # 清空观察状态
            torch_gc()  # 清理显存