"""
2025-05-05 20:14:05 Monday
受不了了，全重写了
新的数据集位置在mydata2，里面是input output process三个子文件夹
数据流:
wiki -> statement -> qa_statement(75) 
conversations -> summary -> qa_summary(25) 写在qa_summary和qa_statement里面了
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

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-uOftWQfqs2MwIAJfiwTPbqMFT8qAJqEWeWOFxC0MZVui10If",
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.org/v1"
)

WIKI_PATH = "/data/hfc/mydata/input/wiki"
GENERAL_PATH = "/data/hfc/mydata/input/general"
PROCESS_PATH_BASE = "/data/hfc/mydata2/process"
OUTPUT_PATH_STATEMENT = f"{PROCESS_PATH_BASE}/statement"

def fill_in_relevant_query_role_template(character, statement, general):
    return f'''已知关于{character}的背景信息：

{general}

人设陈述：{statement}

你需要对{character}提出一些问题，这些问题需要包含上述人设陈述中的信息进行回应。不需要涉及背景信息，保持不违背即可。

提供3个多样且简洁的可能话语，这些话语将谈话对象视为{character}，且不包含名字。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- 你是{character}吗
- 你对你的这一段经历有什么看法'''



def generate_role_queries_v3(role, model_engine):
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{role}_statement.json"
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    

    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()

    if os.path.exists(recall_path):
        print(f"{recall_path} 已存在，跳过生成步骤。")
        return

    os.makedirs(os.path.dirname(with_query_path), exist_ok=True)
    os.makedirs(os.path.dirname(recall_path), exist_ok=True)

    statements_data = load_json(statement_path)
    all_queries = []  # 用于存储所有 query-recall 对

    for item in tqdm(statements_data, desc="Generating Role Queries"):
        for statement in item["statements"]:
            messages = [
                {"role": "system", "content": "你是一个语言改写助手，帮助用户从陈述构筑问题。"},
                {"role": "user", "content": fill_in_relevant_query_role_template(role, statement, general)}
            ]

            # 使用 gpt-4o-mini 生成 queries
            response = client.chat.completions.create(
                model=model_engine,
                messages=messages,
                temperature=1.0
            )
            generated_content = response.choices[0].message.content.strip()

            print("messages:", messages)
            print("generated_content: ", generated_content)

            queries = [q.strip() for q in generated_content.split('\n') if q.startswith('- ')]
            for query in queries:
                all_queries.append({"query": query, "recall": statement})

    # 在 shuffle 之前保存到文件
    with open(with_query_path, 'w', encoding='utf-8') as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=4)

    # 随机组合 query-recall 对
    results = []
    random.shuffle(all_queries)
    for i in range(0, len(all_queries), 2):
        group = all_queries[i:i + random.choice([2, 3])]
        combined_query = ''.join([item["query"] for item in group])
        combined_recall = ''.join([item["recall"] for item in group])
        results.append({
            "instruction": fill_in_relevant_query_role_instruction_template(role),
            "query": combined_query.replace('- ', ''),
            "recall": combined_recall.replace('- ', '')
        })

    save_jsonl(results, recall_path)


generate_role_statements("刘星", "gpt-4o-mini")