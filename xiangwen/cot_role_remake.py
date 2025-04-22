import transformers
import torch
import re
import requests
import json

RECALL_PROMPT_TEMPALTE="""下面是一段关于{world}中的{role}的问题，请为我提供用于回复这些问题的信息。

按问题涉及的信息不同，提供数个多样且简洁的可能信息。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。

示例输出格式：
- {world}是……
- {role}看待……
"""

COT_PROMPT_TEMPALTE="""你正在扮演{world}中的{role}，关于{role}有陈述

{statement}
注意，陈述不一定与问题相关。请尊重事实，以{role}的身份回答问题。需要给出推理过程和最终答案。

需要回答的问题：
"""

class OllamaChat:
    def __init__(self, model='default-model'):
        """
        初始化 OllamaChat 类
        :param model: 使用的模型名称
        """
        self.model = model
        self.api_url = "http://127.0.0.1:11434/api/generate"

    def chat(self, question, history=None, context=None):
        """
        调用本地 Ollama API 生成回答
        :param question: 用户的问题
        :param history: 聊天历史（可选）
        :param context: 上下文信息（可选）
        :return: Ollama 模型的回答
        """
        if history is None:
            history = []

        payload = {
            "model": self.model,
            "prompt": self._build_prompt(question, history, context)
        }
        print("Payload:", payload)  # 打印请求负载
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            # 打印原始响应内容
            # print("Raw response:", response.text)

            # 手动解析多行 JSON 响应
            responses = response.text.strip().split("\n")
            final_response = ""
            for line in responses:
                try:
                    json_line = json.loads(line)
                    if "response" in json_line and json_line["response"]:
                        final_response += json_line["response"]
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line}, error: {e}")

            return final_response if final_response else "No response from Ollama."
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama API: {e}"
        except ValueError as e:
            return f"Error parsing JSON response: {e}"

    def _build_prompt(self, question, history, context):
        """
        构建用于 Ollama API 的提示
        :param question: 用户的问题
        :param history: 聊天历史
        :param context: 上下文信息
        :return: 构建的提示字符串
        """
        prompt = ""
        if context:
            prompt += f"Context: {context}\n\n"
        if history:
            for i, (q, a) in enumerate(history):
                prompt += f"Q{i + 1}: {q}\nA{i + 1}: {a}\n"
        prompt += f"Q: {question}\nA:"
        return prompt
    
recall_chat = OllamaChat(model='liuxing-recall')

cot_chat = OllamaChat(model='liuxing-cot')


# 主循环
print("=== 刘星两阶段语气回答系统 ===")
print("输入示例：小雨：哥，你为什么突然想交女朋友？")
print("输入 q 或 exit 退出。\n")

while True:
    user_input = input("请输入问题 > ").strip()
    if user_input.lower() in {"q", "exit"}:
        break

    # 阶段1：Recall 补充信息
    recall_prompt = RECALL_PROMPT_TEMPALTE.format(
        world="家有儿女",
        role="刘星"
    )
    recall_context = recall_chat.chat(recall_prompt + user_input)
    print("\n[Recall 阶段补充信息]:")
    print(recall_context)

    # 阶段2：CoT 答案生成
    cot_prompt = COT_PROMPT_TEMPALTE.format(
        world="家有儿女",
        role="刘星",
        statement=recall_context
    )
    cot_prompt = "你正在扮演刘星，请以刘星的身份回答问题\n"
    cot_response = cot_chat.chat(cot_prompt + user_input)
    print("\n[CoT 阶段回答]:")
    print(cot_response)