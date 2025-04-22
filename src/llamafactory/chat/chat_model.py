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

# modified feice Apr 19, 2025 at 20:04
def run_cot() -> None:
    if os.name != "nt":
        try:
            import readline  # noqa: F401
        except ImportError:
            print("Install `readline` for a better experience.")

    # 初始化三个模型
    print("正在加载模型...")
    recall_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/Llama-3.1-8B-Instruct",
        "adapter_name_or_path": "/data/hfc/RoleRAG/saves/刘星_recall_800/llama3_8b_sft_lora/TorchTrainer_e8f93_00000_0_2025-04-18_13-13-48/checkpoint_000001/checkpoint"
    })
    cot_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/Llama-3.1-8B-Instruct",
        "adapter_name_or_path": "/data/hfc/RoleRAG/saves/刘星_cot_800/llama3_8b_sft_lora/TorchTrainer_8d771_00000_0_2025-04-17_18-20-14/checkpoint_000003/checkpoint"
    })
    style_model = ChatModel({
        "model_name_or_path": "/data/hfc/checkpoints/Llama-3.1-8B-Instruct",
        "adapter_name_or_path": "/data/hfc/RoleRAG/saves/刘星_style/llama3_8b_sft_lora/TorchTrainer_9fdfa_00000_0_2025-04-16_18-36-15/checkpoint_000000/checkpoint"
    })
    print("模型加载完成！")

    # 定义每个模型的固定前缀
    recall_prefix = "下面是一段关于家有儿女和刘星的问题，请为我提供用于回复这些问题的信息。\n\n按问题涉及的信息不同，提供数个多样且简洁的可能信息。严格遵循示例中的格式，不需要多余分析，避免诸如\"以下是答案：\"之类的陈述。\n示例输出格式：\n\n……\n…\n\n问题："
    cot_prefix = "你正在扮演刘星，请以刘星的身份回答问题\n\n问题："
    additional = "\n\n可能的参考信息："
    style_prefix = "你正在扮演 刘星，你需要将下面的句子转写成 刘星 的口吻\n"

    # 定义问题列表
    questions = [
        "什么是人工智能？",
        "刘星为什么喜欢捣蛋？"
    ]

    for question in questions:
        print(f"\nUser: {question}")

        # 阶段 1：Recall 模型
        print("recall_prefix + question: ", recall_prefix + question)
        recall_messages = [{"role": "user", "content": recall_prefix + question}]
        print("Recall Assistant: ", end="", flush=True)
        recall_response = ""
        for new_text in recall_model.stream_chat(recall_messages):
            print(new_text, end="", flush=True)
            recall_response += new_text
        print("\n"+"="*20)
        torch_gc()  # 清理显存

        # 阶段 2：CoT 模型
        print("cot_prefix + recall_response: ", cot_prefix + question + additional + recall_response)
        cot_messages = [{"role": "user", "content": cot_prefix + question + additional + recall_response}]
        print("CoT Assistant: ", end="", flush=True)
        cot_response = ""
        for new_text in cot_model.stream_chat(cot_messages):
            print(new_text, end="", flush=True)
            cot_response += new_text
        print("\n"+"="*20)
        torch_gc()  # 清理显存

        # 阶段 3：Style 模型
        cot_last_response = cot_response.split("\n")[-1].strip()
        print("style_prefix + cot_last_response: ", style_prefix + cot_last_response)
        style_messages = [{"role": "user", "content": style_prefix + cot_last_response}]
        # style_messages = [{"role": "user", "content": style_prefix + cot_response}]
        print("Style Assistant: ", end="", flush=True)
        style_response = ""
        for new_text in style_model.stream_chat(style_messages):
            print(new_text, end="", flush=True)
            style_response += new_text
        print("\n"+"="*20)
        torch_gc()  # 清理显存

        print("Final Output: ", style_response)
        print("History has been removed.\n")