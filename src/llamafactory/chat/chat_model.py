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

# modified feice May 8, 2025 at 18:26
实现大规模处理

"""
import asyncio
import os
import json
from collections.abc import AsyncGenerator, Generator
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional
from tqdm import tqdm 

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
# import requests
# import time

# GOOGLE_API_URL = "https://www.googleapis.com/customsearch/v1"
# GOOGLE_API_KEY = "AIzaSyDO3hlUuK9CugxCIQf86czKk-_K_NeHvuA"  # 替换为你的 Google API 密钥
# GOOGLE_CX = "87b42182cc13c4b23"  # 替换为你的 Google Custom Search Engine ID

# def google_search(query, num_results=4, timeout=5):
#     """
#     使用 Google Custom Search JSON API 执行查询并返回结果。

#     :param query: 搜索的关键词或主题
#     :param num_results: 返回的搜索结果数量（默认 4）
#     :param timeout: 请求超时时间（默认 5 秒）
#     :return: 检索到的文本和搜索耗时
#     """
#     params = {
#         "q": query,
#         "key": GOOGLE_API_KEY,
#         "cx": GOOGLE_CX,
#         "num": num_results
#     }

#     # 设置代理
#     proxies = {
#         "http": "http://219.223.184.164:7890",
#         "https": "http://219.223.184.164:7890"
#     }

#     try:
#         start_time = time.time()
#         response = requests.get(GOOGLE_API_URL, params=params, timeout=timeout, proxies=proxies)
#         response.raise_for_status()  # 如果响应状态码不是 200，抛出异常
#         search_time = time.time() - start_time

#         # 解析响应 JSON
#         response_json = response.json()
#         items = response_json.get("items", [])
        
#         # 提取搜索结果
#         results = []
#         for item in items:
#             title = item.get("title", "")
#             snippet = item.get("snippet", "")
#             results.append(f"{title}: {snippet}")
        
#         # 将结果合并为单个字符串
#         result_text = " ".join(results)
#         return result_text, search_time

#     except requests.exceptions.RequestException as e:
#         print(f"Error during Google search: {e}")
#         return "", 0

# 定义每个任务的system以及instruction prompt
def get_rewrite_prompt(question: str) -> str:
    return f"""下面是一段关于家有儿女和刘星的问题，这段问题将“你”视为“刘星”。
请为我将问题转化为适用于RAG检索的陈述。
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

def get_style_prompt_cot(role, question: str) -> str:
    return f"""你正在扮演 {role}，你需要将下面的句子转写成 {role} 的口吻
{question}
"""

# modified feice May 8, 2025 at 16:58
def run_cot() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")
    # 参数
    role = "刘星"
    world = "家有儿女"
    model_name = "cot_glm4_glm4_glm4"

    # 初始化三个模型
    print("正在加载模型...")
    rewrite_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/GLM-4-9B-0414",
        "template": "glm4"
    })
    cot_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/GLM-4-9B-0414",
        "adapter_name_or_path": "/data/hfc/RoleRAG/saves/刘星_glm4_cot_0508/glm4_8b_sft_lora/TorchTrainer_67654_00000_0_2025-05-08_16-57-00/checkpoint_000001/checkpoint",
        "template": "glm4"
    })
    style_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/GLM-4-9B-0414",
        "adapter_name_or_path": "/data/hfc/RoleRAG/saves/刘星_glm4_style/glm4_8b_sft_lora/TorchTrainer_8bff6_00000_0_2025-05-08_20-32-46/checkpoint_000001/checkpoint/",
        "template": "glm4"
    })
    # 初始化
    retriever = Retriever(config={
        "embedding_model": "/data/hfc/checkpoints/text2vec-large-chinese"
    })

    print("模型加载完成！")

    

    # 1. 读取原始json
    qa_path = "/data/hfc/RoleRAG/data0506/all/家有儿女_刘星_qa_135.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # 2. 检索相关文档
    file_paths = [
        f"/data/hfc/RoleRAG/data0506/input/wiki/wiki_{role}.txt",
        f"/data/hfc/RoleRAG/data0506/input/wiki/wiki_{world}.txt",
        f"/data/hfc/RoleRAG/data0506/process/summary/{world}_{role}_summary.json"
    ]
    docs = retriever.load_files(file_paths)
    retriever.create_vector_store(docs)
    retriever.save_vector_store(f"/data/hfc/faiss_store/{world}_{role}")

    # 3. 处理每个问题
    for item in tqdm(qa_data, desc="CoT处理中"):
        print("question: ", item.get("question", "").strip())
        question = item.get("question", "").strip()
        if not question:
            continue

        rewrite_messages = [{"role": "user", "content": get_rewrite_prompt(question)}]
        rewrite_response = ""
        for new_text in rewrite_model.stream_chat(
            rewrite_messages,
            system="你是一个RAG模型的重写器，请将问题转化为适用于RAG检索的陈述"
        ):
            rewrite_response += new_text
        print("rewrite_response: ", rewrite_response)
        torch_gc()



        # 检索 observation
        retrieved_docs = retriever.retrieve_top_k(rewrite_response, k=3, with_context=False)
        observation = ""
        for doc in retrieved_docs:
            observation += doc.page_content + "\n"
        print("observation: ", observation)

        # 阶段 2：CoT 模型
        cot_messages = [{"role": "user", "content": get_cot_prompt(question, observation)}]
        cot_response = ""
        for new_text in cot_model.stream_chat(
            cot_messages,
            system="你是一个角色扮演专家，请以【问题重述】【实体确认】【逻辑推理】【分析回答】【最终回答】的顺序，以扮演角色的身份回答问题"
        ):
            cot_response += new_text
        print("cot_response: ", cot_response)
        torch_gc()

        # 阶段 3：Style 模型
        cot_last_response = cot_response.split("\n")[-1].strip()
        print("get_style_prompt: ", get_style_prompt_cot(role, cot_last_response))
        style_messages = [{"role": "user", "content": get_style_prompt_cot(role, cot_last_response)}]
        style_response = ""
        for new_text in style_model.stream_chat(
            style_messages,
            system="你是一个语言改写助手，将这段语句转换为扮演人物的说话语气。"
        ):
            style_response += new_text
        print("style_response: ", style_response)
        torch_gc()

        # 合并结果到item
        item["observation"] = observation
        item["cot_response"] = cot_response
        item["style_response"] = style_response

    # 4. 保存结果
    output_dir = "/data/hfc/RoleRAG/output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{world}_{role}_{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    print(f"已保存到 {output_path}")



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
已有的参考信息：
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
已有的参考信息：
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
    
# modified feice May 8, 2025 at 17:08
def run_rag() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    # 设置角色与世界
    # 1. 设置角色、世界、数据集路径、模型名、检索数据库路径
    role = "刘星"
    world = "家有儿女"
    model_name = "rag"
    qa_path = "/data/hfc/RoleRAG/data0506/all/家有儿女_刘星_qa_135.json"
    faiss_store_path = f"/data/hfc/faiss_store/{world}_{role}_rag"
    output_dir = "/data/hfc/RoleRAG/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{world}_{role}_{model_name}.json")
    profile = "刘星，刘梅之子。初中生（在第四部升上高中生），成绩（尤其化学）常令刘梅头痛。身材看似“瘦弱”，体育倒很不错。爱好广泛但大多都只折腾一时。一家的活宝，大多数麻烦的制造者。为人仗义，脑子里经常有些新奇的想法，里面有好主意也有馊主意。"


    # 2. 初始化模型和检索器
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
    file_paths = [
        f"/data/hfc/RoleRAG/data0506/input/wiki/wiki_{role}.txt",
        f"/data/hfc/RoleRAG/data0506/input/wiki/wiki_{world}.txt",
        f"/data/hfc/RoleRAG/data0506/process/summary/{world}_{role}_summary.json"
    ]

    # 加载文件并创建向量数据库
    docs = retriever.load_files(file_paths)
    retriever.create_vector_store(docs)

    # 保存向量数据库
    retriever.save_vector_store(faiss_store_path)

    # 3. 读取数据集
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    # 4. 遍历每个问题
    for item in qa_data:
        question = item.get("question", "").strip()
        print("question: ", question)
        if not question:
            continue
        observation = ""
        max_attempts = 5
        attempts = 0
        model_response = ""
        while attempts < max_attempts:
            attempts += 1
            # 4.1 生成RAG prompt并调用模型
            rag_prompt = get_rag_prompt(role, profile, question, observation)
            messages = [{"role": "user", "content": rag_prompt}]
            model_response = ""
            for new_text in rag_model.stream_chat(messages):
                model_response += new_text
            model_response = model_response.lstrip("\n").lstrip(" ")
            print("model_response: ", model_response)
            # 4.2 判断是否需要检索
            if model_response.startswith("[Retrieve]"):
                # 检索并更新observation
                retrieved_docs = retriever.retrieve_top_k(question, k=1, with_context=False)
                retrieval_text = "\n".join([doc.page_content for doc in retrieved_docs])
                observation += model_response + ": " + retrieval_text
            elif model_response.startswith("[Answer]"):
                break
            else:
                # 无效响应，直接跳出
                break

        # 5. 得到答案或最大次数后调用final_answer
        if not model_response.startswith("[Answer]"):
            print("observation: ", observation)
            final_prompt = get_final_answer_prompt(role, profile, question, observation)
            final_answer = ""
            for new_text in rag_model.stream_chat([{"role": "user", "content": final_prompt}]):
                final_answer += new_text
            model_response = final_answer.lstrip("\n").lstrip(" ")

        # 6. 解析答案，进入风格化
        answer = model_response[len("[Answer]"):].strip() if model_response.startswith("[Answer]") else model_response.strip()
        style_prompt = get_style_prompt(role, answer)
        style_response = ""
        for new_text in rag_model.stream_chat([{"role": "user", "content": style_prompt}]):
            style_response += new_text
        style_answer = style_response[len("[Style]"):].strip() if style_response.startswith("[Style]") else style_response.strip()
        print("style_answer: ", style_answer)

        # 7. 保存到item
        item["observation"] = observation
        item["final_answer"] = answer
        item["style_response"] = style_answer

        torch_gc()

    # 8. 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    print(f"已保存到 {output_path}")