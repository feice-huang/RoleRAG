"""
** wiki -> statement -> qa_statement(75) 
conversations -> summary -> qa_summary(25)
GPT -> qa_chat(25)
GPT -> qa_leading(75)
"""

import glob
import os
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
import requests
from openai import OpenAI


WIKI_PATH = "/data/hfc/RoleRAG/data0506/input/wiki"
GENERAL_PATH = "/data/hfc/RoleRAG/data0506/input/general"
PROCESS_PATH_BASE = "/data/hfc/RoleRAG/data0506/process"
OUTPUT_PATH_STATEMENT = f"{PROCESS_PATH_BASE}/statement"

def fill_in_role_statement_template(character, passage, general):
    return f'''已知关于{character}的背景信息：

{general}

给定关于"{character}"的段落：

{passage}

请根据这个段落，生成一些关于"{character}"的重要人设陈述，供角色扮演的AI遵循。

- 严格遵循以下格式，每个陈述以"- "开头。
- 确保每个陈述中明确提到"{character}"，避免使用代词或共指。
- 尽可能保留段落中的信息，特别是书名、地点、生日或组织等实体细节。
- 指出事实，而不要给出空洞的话语，避免使用"模式"、"特性"、"范式"类型的词语。
- 仅关注给定的段落，不要引用历史对话中的信息，不要引用背景信息，仅保持不违背即可。
- 避免生成诸如"以下是答案："之类的介绍性短语。

示例输出格式：

- {character}是...
- {character}有...
- {character}主要研究...
- {character}的作品包括...

生成陈述时严格遵循这些指示。
'''

def load_wiki_data(role):
    wiki_file = f"{WIKI_PATH}/wiki_{role}.txt"
    with open(wiki_file, 'r', encoding='utf-8') as f:
        return [p.strip() for p in f.read().split('\n\n') if p.strip()]

def wiki2statement(role, apikey, model_engine="gpt-4o-mini"):
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=apikey,
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.org/v1"
)
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    # 读取 general 数据, txt里面只有一行
    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{role}_statement.json"
    if os.path.exists(statement_path):
        print(f"{statement_path} 已存在，跳过生成步骤。")
        return

    os.makedirs(os.path.dirname(statement_path), exist_ok=True)
    wiki_data = load_wiki_data(role)
    results = []

    for passage in tqdm(wiki_data, desc="Generating Statements"):
        messages = [
            {"role": "system", "content": "你是一个语言改写助手，帮助用户从段落构筑陈述。"},
            {"role": "user", "content": fill_in_role_statement_template(role, passage, general)}
        ]

        try:
            # 使用 gpt-4o-mini 模型进行生成
            response = client.chat.completions.create(
                model=model_engine,
                messages=messages,
                temperature=1.0
            )
            
            # 检查返回内容是否为空
            content = response.choices[0].message.content
            generated_content = content.strip() if content else "无有效响应"
            
            print("messages:", messages)
            print("generated_content: ", generated_content)

            # 只有在有内容时才尝试提取陈述
            if generated_content != "无有效响应":
                statements = [s.strip().replace('- ', "") for s in generated_content.split('\n') if s.startswith('- ')]
                results.append({"passage": passage, "statements": statements})
            else:
                print(f"警告: 段落 '{passage[:30]}...' 没有生成有效陈述，已跳过")
                # 可以选择添加空陈述或重试
                results.append({"passage": passage, "statements": []})
            
            # 添加一个短暂的延迟，避免频繁请求API
            time.sleep(1)
            
        except Exception as e:
            print(f"处理段落时出错: {str(e)}")
            print(f"出错的段落: '{passage[:50]}...'")
            # 添加空陈述并继续
            results.append({"passage": passage, "statements": []})
            time.sleep(3)  # 遇到错误时等待更长时间

    with open(statement_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"生成完成！已保存到: {statement_path}")


if __name__ == "__main__":
    # 示例用法
    apikey = "sk-MA7hKS37UdRUmP3Xz4BzHt3Rqj6QFbRoEagxcmFwwBBHyZR6"
    # 生成角色陈述
    wiki2statement("刘星", apikey, "gpt-4o-mini")