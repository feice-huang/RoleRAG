"""
生成角色扮演的思维链数据集
代码直接设置参数，无需命令行解析
"""
import glob
import os
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
import requests

from RoleRAG.data_generate.wiki2statement import load_json

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-uOftWQfqs2MwIAJfiwTPbqMFT8qAJqEWeWOFxC0MZVui10If",
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.org/v1"
)

def load_wiki_data(role):
    wiki_file = f"{WIKI_PATH}/wiki_{role}.txt"
    with open(wiki_file, 'r', encoding='utf-8') as f:
        return [p.strip() for p in f.read().split('\n\n') if p.strip()]

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

def generate_role_statements(role, model_engine, token):
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
        _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
        print("messages:", messages)
        print("generated_content: ", generated_content)
        statements = [s.strip() for s in generated_content.split('\n') if s.startswith('- ')]
        results.append({"passage": passage, "statements": statements})

    with open(statement_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    # save_jsonl(results, statement_path)


if __name__ == "__main__":
    # 参数设置
    CHARACTER = "刘星"
    MODEL_NAME = "gpt-4o-mini"  # 使用OpenAI的4o-mini模型
    
    # 文件路径
    input_file = f"/data/hfc/RoleRAG/mydata/summary/家有儿女_{CHARACTER}_summary.json"
    output_dir = "/data/hfc/RoleRAG/mydata/qa"
    output_file = os.path.join(output_dir, f"家有儿女_{CHARACTER}_qa.jsonl")
    
    # 生成问答对
    generate_qa_for_character(input_file, CHARACTER, MODEL_NAME, output_file)