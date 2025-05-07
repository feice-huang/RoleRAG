import glob
import os
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
import requests
from openai import OpenAI
import re

def chat2qa(world, role, api_key, model_engine="gpt-4o-mini"):
    """
    封装整个文件的功能为一个函数。
    :param world: 世界名称（如"家有儿女"）
    :param role: 角色名称
    :param api_key: OpenAI API 密钥
    :param model_engine: GPT模型名称
    """
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.chatanywhere.tech/v1"
    )

    # 路径配置
    GENERAL_PATH = "/data/hfc/RoleRAG/data0506/input/general"
    OUTPUT_PATH_QA_CHAT = "/data/hfc/RoleRAG/data0506/qa/qa_chat"

    def load_json(file_path):
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_topics_template(character, general):
        """
        生成聊天主题的模板
        """
        return f'''已知关于{character}的背景信息：

{general}

请基于上述背景信息，生成10个与{character}相关的聊天主题。这些主题需要多样化且与{character}的背景信息相关。严格遵循示例中的格式(str数组)，不需要多余分析。

示例输出格式：
[
    "主题1",
    "主题2",
    "主题3",
    ...
    "主题10"
]
'''

    def generate_qa_template(character, general, topic):
        """
        基于主题生成问答对的模板
        """
        return f'''已知关于{character}的背景信息：

{general}

聊天主题：{topic}

请基于上述主题，生成3个多样且简洁的问题-回答对。这些问题将谈话对象视为{character}，且不包含名字；回答者以{character}的身份回答。严格遵循示例中的格式(json数组)，不需要多余分析。

示例输出格式：
[
    {{
        "question": "问题1",
        "answer": "回答1"
    }},
    {{
        "question": "问题2",
        "answer": "回答2"
    }},
    {{
        "question": "问题3",
        "answer": "回答3"
    }}
]
'''

    # 确保输出目录存在
    os.makedirs(OUTPUT_PATH_QA_CHAT, exist_ok=True)

    # 读取通用背景信息
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()

    # 用于存储生成的问答对
    all_qa_pairs = []

    # Step 1: 生成聊天主题
    messages = [
        {"role": "system", "content": "你是一个语言改写助手，帮助用户从背景信息生成聊天主题。"},
        {"role": "user", "content": generate_topics_template(role, general)}
    ]

    max_retries = 3
    retry_count = 0
    topics = []

    while retry_count < max_retries:
        try:
            # 调用 GPT 模型生成聊天主题
            response = client.chat.completions.create(
                model=model_engine,
                messages=messages,
                temperature=0.8,
                timeout=60  # 60秒超时
            )
            generated_content = response.choices[0].message.content.strip()

            print(f"--- 生成主题内容: {generated_content[:100]}...")

            # 解析JSON格式的响应
            try:
                topics = json.loads(generated_content)
                if not isinstance(topics, list):
                    print(f"警告: 预期列表但得到 {type(topics)}")
                    retry_count += 1
                    continue
                break  # 成功解析，退出重试循环
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"原始内容: {generated_content}")
                retry_count += 1
                continue

        except Exception as e:
            print(f"API调用错误: {e}")
            time.sleep(5 * (retry_count + 1))  # 指数退避
            retry_count += 1
            continue

    if not topics:
        print("未能生成聊天主题，退出程序。")
        return

    # Step 2: 基于每个主题生成问答对
    for topic_index, topic in enumerate(topics):
        print(f"\n--- 处理主题 [{topic_index+1}/{len(topics)}]: {topic} ---")
        messages = [
            {"role": "system", "content": "你是一个语言改写助手，帮助用户从聊天主题生成问答对。"},
            {"role": "user", "content": generate_qa_template(role, general, topic)}
        ]

        retry_count = 0
        while retry_count < max_retries:
            try:
                # 调用 GPT 模型生成问答对
                response = client.chat.completions.create(
                    model=model_engine,
                    messages=messages,
                    temperature=0.8,
                    timeout=60  # 60秒超时
                )
                generated_content = response.choices[0].message.content.strip()

                print(f"--- 生成问答内容: {generated_content[:100]}...")

                # 解析JSON格式的响应
                try:
                    qa_pairs = json.loads(generated_content)
                    if not isinstance(qa_pairs, list):
                        print(f"警告: 预期列表但得到 {type(qa_pairs)}")
                        retry_count += 1
                        continue

                    # 添加 retrieve 字段并存储
                    for qa_pair in qa_pairs:
                        if "question" in qa_pair and "answer" in qa_pair:
                            qa_pair["retrieve"] = ""
                            all_qa_pairs.append(qa_pair)
                    break  # 成功解析，退出重试循环
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    print(f"原始内容: {generated_content}")
                    retry_count += 1
                    continue

            except Exception as e:
                print(f"API调用错误: {e}")
                time.sleep(5 * (retry_count + 1))  # 指数退避
                retry_count += 1
                continue

    # 根据实际生成的问答对数量动态命名文件
    actual_count = len(all_qa_pairs)
    output_path = f"{OUTPUT_PATH_QA_CHAT}/{world}_{role}_qa_{actual_count}.json"

    # 保存生成的问答对
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
    print(f"已生成 {actual_count} 个问答对，保存到 {output_path}")