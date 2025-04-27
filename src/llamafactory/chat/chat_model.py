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

"""
import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from threading import Thread
from typing import TYPE_CHECKING, Any, Optional

from ..extras.constants import EngineName
from ..extras.misc import torch_gc
from ..hparams import get_infer_args
from .hf_engine import HuggingfaceEngine
from .sglang_engine import SGLangEngine
from .vllm_engine import VllmEngine


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
    return f"""你正在扮演刘星，请以刘星的身份回答问题
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
        "model_name_or_path": "/data/hfc/checkpoints/Llama-3.1-8B-Instruct",
        "adapter_name_or_path": "/data/hfc/RoleRAG/saves/刘星_style/llama3_8b_sft_lora/TorchTrainer_9fdfa_00000_0_2025-04-16_18-36-15/checkpoint_000000/checkpoint",
        "template": "llama3"
    })
    print("模型加载完成！")

    # 定义问题列表
    questions = [
        "你如何看待人工智能？",
        "你为什么喜欢捣蛋？"
    ]

    for question in questions:
        print(f"\nUser: {question}")

        # 阶段 1：Recall 模型
        print("recall instruction: ", get_recall_prompt(question))
        recall_messages = [{"role": "user", "content": get_recall_prompt(question)}]
        print("Recall Assistant: ", end="", flush=True)
        observation = ""
        for new_text in rewrite_model.stream_chat(recall_messages, system="你是一个人工智能问题重写助手，按照规则重写问题"):
            print(new_text, end="", flush=True)
            observation += new_text
        print("\n"+"="*20)
        torch_gc()  # 清理显存
        ans, _ = google_search(observation)
        observation = observation + "\n" + ans
        # observation = "刘星，刘梅之子。初中生（在第四部升上高中生），成绩（尤其化学）常令刘梅头痛。身材看似“瘦弱”，体育倒很不错。爱好广泛但大多都只折腾一时。一家的活宝，大多数麻烦的制造者。为人仗义，脑子里经常有些新奇的想法，里面有好主意也有馊主意。"



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
def get_score_prompt(question: str) -> str:
    return f"""按照下面的规则为问题进行难度打分（越高越难）：
1. 涉及人物：问题包含了几个人物（0：不涉及人物，1:仅涉及扮演角色，2:涉及2个角色，3：涉及3个及以上的角色）
2. 涉及时间：问题涉及了多么精确的时间（0:不涉及时间，1:涉及大范围时间（年代），2:涉及年份，3:涉及具体事件，4：涉及具体对话上下文）
3. 问题相关性：问题和扮演角色有多么相关（0：不需要了解扮演角色即可回答，1：需要结合角色身份回答，2：需要结合角色性格回答，3：需要结合角色具体事件回答）
回答格式为[score] <分数>，例如：[score] (2+4+3) <9>。注意：分数范围是0-10分，0分表示问题不涉及角色扮演，10分表示问题非常复杂。
示例：
[score] (2+4+3) <9>

你需要打分的问题是：
{question}
"""

def get_search_prompt(question: str) -> str:
    return f"""将下面的问题重写为适合检索的问题，注意：
1. 将问题中的任何代词都改成明确的名称，2. 不要包含任何角色扮演的提示，3. 不要包含任何上下文信息，4. 不要包含任何多余的分析和解释。
回答格式为[search] <问题>
示例：[search] <刘星喜欢什么>。

你需要重写的问题是：
{question}
"""
    

def get_answer_prompt(role: str, profile: str, question: str, observation: str) -> str:
    return f"""根据下面的角色扮演信息，判断信息是否足够回答问题，如果信息不足，需要给出用于解决问题的辅助子问题，子问题需要非常简单，专注于关键信息。
如果信息足以回答，则输出:
[answer] <回答内容>
如果信息不足以回答，则输出：
[subquery] <字问题1> <子问题2>
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
[answer] <回答内容>
"""


def get_style_prompt(role: str, sentence: str) -> str:
    return f"""你需要将下面的句子转写成对应角色的口吻，注意只需要给出答案，不需要任何解释和分析。
你需要扮演的角色是：
{role}
你需要转写的句子是：
{sentence}
"""
    
# modified feice Apr 24, 2025 at 21:24
def run_rag() -> None:
    def run_score(model: ChatModel, question: str) -> int:
        """运行 Score 阶段，返回分数"""
        torch_gc()  # 清理显存
        score_prompt = get_score_prompt(question)
        score_messages = [{"role": "user", "content": score_prompt}]
        print("\nScore Prompt:\n", score_prompt)
        score_response = ""
        for new_text in model.stream_chat(score_messages, system="你是一个人工智能打分助手，按照规则为下面的角色扮演相关问题进行难度打分"):
            print(new_text, end="", flush=True)
            score_response += new_text
        print("\n" + "=" * 20)

        # 解析分数
        try:
            score = int(score_response.split("<")[-1].split(">")[0])
        except ValueError:
            print("Score parsing failed, defaulting to 10.")
            score = 10
        return score
    
    def run_search(model: ChatModel, question: str, observation: str) -> str:
        """运行 Search 阶段，返回更新后的 observation"""
        torch_gc()  # 清理显存
        search_prompt = get_search_prompt(question)
        search_messages = [{"role": "user", "content": search_prompt}]
        print("\nSearch Prompt:\n", search_prompt)
        search_response = ""
        for new_text in model.stream_chat(search_messages, system="你是一个人工智能问题重写助手，按照规则将下面的问题进行重写"):
            print(new_text, end="", flush=True)
            search_response += new_text
        print("\n" + "=" * 20)

        # 调用 Google Search API
        search_query = search_response.split("<")[-1].split(">")[0]
        search_result, _ = google_search(search_query)
        observation += f"\n{search_query}: {search_result}"
        return observation

    def run_answer(model: ChatModel, role: str, profile: str, question: str, observation: str, search_count: int, max_search_count: int) -> tuple[Optional[str], Optional[list[str]]]:
        """运行 Answer 阶段，返回答案或子问题"""
        torch_gc()  # 清理显存
        answer_prompt = get_final_answer_prompt(role, profile, question, observation) if search_count > max_search_count else get_answer_prompt(role, profile, question, observation)
        answer_messages = [{"role": "user", "content": answer_prompt}]
        print("\nAnswer Prompt:\n", answer_prompt)
        answer_response = ""
        for new_text in model.stream_chat(answer_messages, system="你是一个角色扮演专家，请根据下面的角色扮演信息回答问题"):
            print(new_text, end="", flush=True)
            answer_response += new_text
        print("\n" + "=" * 20)

        # 判断是否回答出问题
        if "[answer]" in answer_response:
            answer = answer_response.split("[answer]")[-1].strip()
            return answer, None
        elif "[subquery]" in answer_response:
            subqueries = answer_response.split("[subquery]")[-1].strip().split("<")[1:]
            subqueries = [q.split(">")[0] for q in subqueries]
            return None, subqueries
        else:
            return None, None

    def run_style(model: ChatModel, role: str, answer: str) -> str:
        """运行 Style 阶段，返回风格化后的答案"""
        torch_gc()  # 清理显存
        style_prompt = get_style_prompt(role, answer)
        style_messages = [{"role": "user", "content": style_prompt}]
        print("\nStyle Prompt:\n", style_prompt)
        style_response = ""
        for new_text in model.stream_chat(style_messages, system="你是一个角色扮演专家，你需要将下面的句子转写成对应角色的口吻"):
            print(new_text, end="", flush=True)
            style_response += new_text
        print("\n" + "=" * 20)
        return style_response

    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    # 初始化模型
    print("正在加载模型...")
    rag_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/GLM-4-9B-0414",
        "template": "glm4"
    })
    print("模型加载完成！")

    # 定义问题列表
    questions = [
        "你喜欢玩原神吗？",
        "你为什么这么笨"
    ]
    role = "刘星"
    profile = "刘星是一个聪明、机智、幽默的男孩，喜欢捣蛋和恶作剧。他的父母是刘梅和夏东海，他们对他的行为感到无奈，但也很宠爱他。"

    for question in questions:
        print(f"\nUser: {question}")

        # 初始化变量
        observation = ""
        search_count = 0
        max_search_count = 3
        final_answer = None

        # 阶段 1：Score
        score = run_score(rag_model, question)

        # 循环处理，直到回答出问题或达到最大搜索次数
        while not final_answer and search_count <= max_search_count:
            if score >= 4:
                # 阶段 2：Search
                search_count += 1
                observation = run_search(rag_model, question, observation)
            # 阶段 3：Answer
            answer, subqueries = run_answer(rag_model, role, profile, question, observation, search_count, max_search_count)
            if answer:
                final_answer = run_style(rag_model, role, answer)
            elif subqueries:
                question = " ".join(subqueries)
            else:
                print("Answer parsing failed, exiting loop.")
                break

        # 打印最终结果
        if final_answer:
            print("\nFinal Output: ", final_answer)
        else:
            print("Failed to answer the question within the search limit.")


