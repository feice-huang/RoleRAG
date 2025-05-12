import glob
import os
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
import requests

from openai import OpenAI

def anti2qa(world, role, api_key, model_engine="gpt-4o"):
    """
    封装整个反例生成逻辑为一个函数。
    :param world: 世界名称
    :param role: 角色名称
    :param api_key: OpenAI API 密钥
    """
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.chatanywhere.tech/v1"
    )

    # 路径配置
    GENERAL_PATH = "/data/hfc/RoleRAG/data0506/input/general"
    ANTI_PATH = "/data/hfc/RoleRAG/data0506/input/anti"  # 反例关键词的路径
    PROCESS_PATH_BASE = "/data/hfc/RoleRAG/data0506/process"
    OUTPUT_PATH_QA_ANTI = f"/data/hfc/RoleRAG/data0506/qa/qa_anti"

    def load_json(file_path):
        """
        加载 JSON 文件。
        :param file_path: 文件路径
        :return: JSON 数据
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def fill_in_role_anti_template(world, role, question_type, description, keyword, general):
        return f'''已知关于{role}的背景信息：

{general}

你正在对AI扮演的{world}中的{role}进行诱导性提问。

请问出两个{role}不应该回答上来的问题，并给出对应表现出恰当迷茫和不解的回答。
注意，当问题中仅有一部分超出认知时，需要给出一个模糊的回答，而不是完全不知所云。

问题类型：{question_type}
问题描述：{description}
问题关键词：{keyword}

严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：
[
    {{
        "query": "你了解ChatGPT吗？",
        "answer": "那是什么东西？学校的英语课没学过这个单词啊。"
    }},
    {{
        "query": "你用ChatGPT写过作业吗？",
        "answer": "ChatGPT？我没用听说过这个东西呀。"
    }}
]'''

    anti_path = os.path.join(ANTI_PATH, f"anti_{role}.json")
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    qa_anti_path = os.path.join(OUTPUT_PATH_QA_ANTI, f"{role}_with_anti_query.json")

    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()
        
    # 检查文件是否存在
    if os.path.exists(qa_anti_path):
        print(f"{qa_anti_path} 已存在，跳过生成步骤。")
        return

    # 创建目标目录
    os.makedirs(os.path.dirname(qa_anti_path), exist_ok=True)

    # 读取反例关键词数据
    anti_data = load_json(anti_path)
    results = []

    for item in tqdm(anti_data, desc=f"Generating Anti Queries for {role}"):
        question_type = item["type"]
        description = item["description"]
        example_keywords = item["example_keywords"]

        for keyword in example_keywords:
            # 构造提示消息
            messages = [
                {"role": "system", "content": "你是一个语言改写助手，帮助用户生成诱导性问题和回答。"},
                {"role": "user", "content": fill_in_role_anti_template(world, role, question_type, description, keyword, general)}
            ]

            # 使用 gpt-4o 生成内容
            response = client.chat.completions.create(
                model=model_engine,
                messages=messages,
                temperature=1.0
            )
            generated_content = response.choices[0].message.content.strip()

            print("messages:", messages)
            print("generated_content:", generated_content)

            # 解析生成的内容
            try:
                # 将生成的内容解析为 JSON 数组
                parsed_content = json.loads(generated_content)
                for qa in parsed_content:
                    query = qa.get("query")
                    answer = qa.get("answer")
                    if query and answer:
                        results.append({"question": query, "answer": answer, "retrieve": "", "hallucination": question_type})
            except json.JSONDecodeError as e:
                print(f"解析生成内容时出错: {e}")
                print("生成的内容:", generated_content)

    # 根据生成数据的条数动态命名文件
    count = len(results)
    output_file_name = f"{world}_{role}_qa_{count}.json"
    output_file_path = os.path.join(OUTPUT_PATH_QA_ANTI, output_file_name)

    # 保存生成的反例问题和回答
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Saved anti queries to {output_file_path}")

# 示例调用
if __name__ == "__main__":
    anti2qa("家有儿女", "刘星", "sk-MA7hKS37UdRUmP3Xz4BzHt3Rqj6QFbRoEagxcmFwwBBHyZR6")