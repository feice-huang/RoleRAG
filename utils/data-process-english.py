import glob
import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import requests

# 输入数据的路径
DATASET_PATH = "/data/hfc/datasets/RoleAgentBench/家有儿女 S1E1"
WIKI_PATH = "/data/hfc/mydata/wiki"

# 输出数据的路径，最后有用的东西是f"{OUTPUT_PATH_BASE}/{role}_sft_shuffle.jsonl"和f"{OUTPUT_PATH_BASE}/{role}_dpo_shuffle.jsonl"，分别是sft和dpo的数据
OUTPUT_PATH_BASE = "/data/hfc/mydata"

# 输出数据的路径，用于存储中间数据
OUTPUT_PATH_SFT = f"{OUTPUT_PATH_BASE}/sft" # 四个中间文件，general_response, reaction, self_knowledge, summary
OUTPUT_PATH_DPO = f"{OUTPUT_PATH_BASE}/dpo" # 两个中间文件，conversation和generated_qa
OUTPUT_PATH_STATEMENT = f"{OUTPUT_PATH_BASE}/statement" # 一个中间文件，statement
OUTPUT_PATH_WITH_QUERY = f"{OUTPUT_PATH_BASE}/with_query" # 一个中间文件，with_query
OUTPUT_PATH_CHOSEN = f"{OUTPUT_PATH_BASE}/chosen" # 一个中间文件，chosen


def call_tsinghua_deepseek(model, token, messages):
    """
    调用清华大学的 DeepSeek 模型
    :param model: "DeepSeek-R1-671B"
    :param token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0MjI2OTkyOCwiZXhwIjoxNzQyMjkxNTI4fQ.7K6CaJ0I7O4pYzAdJHriuCeRaehREL-o8VZjFYrvnOk"
    :param messages: "你好"
    :return: think, response
    """
    url = 'https://madmodel.cs.tsinghua.edu.cn/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "stream": True
    }

    buffer = ""  # 缓冲池
    full_content = ""  # 用于累积所有内容

    try:
        with requests.post(url, headers=headers, json=data, stream=True) as r:
            # print(f"请求状态码: {r.status_code}")
            if r.status_code != 200:
                print(f"请求失败，状态码: {r.status_code}, 响应: {r.text}")
            else:
                # 逐行读取响应流
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        # print(f"接收到的行: {decoded_line}")
                        if decoded_line.startswith("data: "):
                            chunk = decoded_line[6:]  # 去掉 'data: ' 前缀
                            buffer += chunk  # 加入缓冲池

                            # 尝试解码
                            try:
                                json_data = json.loads(buffer)
                                content = json_data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    full_content += content  # 累积内容
                                buffer = ""  # 解码成功，清空缓冲池
                            except json.JSONDecodeError:
                                # 数据不完整，继续累积
                                pass

                        # 结束标志
                        if decoded_line == "data: [DONE]":
                            break

    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")

    # 在最后进行分割
    if "</think>" in full_content:
        think, response = full_content.split("</think>")
        return think.strip(), response.strip()
    else:
        return "", full_content.strip()


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_conversation(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data, file_path):
    """
    保存数据到 JSONL 文件
    :param data: 数据列表，json数组
    :param file_path: 文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---- SFT 数据转换 ----
def convert_general_response(role):
    data = load_json(f"{DATASET_PATH}/general_response.json")
    output = [
        {
            "instruction": f"You are playing the role of {role}, answer the question as {role}.",
            "input": f"{entry['source_role']}：{entry['question']}",
            "output": entry["answer"]
        }
        for entry in data if entry["target_role"] == role
    ]
    save_jsonl(output, f"{OUTPUT_PATH_SFT}/{role}_general_response.jsonl")


def convert_reaction(role):
    data = load_json(f"{DATASET_PATH}/reaction.json")
    output = [
        {
            "instruction": f"You are playing the role of {role}, answer the question as {role}.",
            "input": f"{entry['source_role']}：{entry['question']}，choose the answer from the options below and directly respond with the ABCD: {entry['multi_choices']}",
            "output": entry["gt_answer"]
        }
        for entry in data if entry["target_role"] == role
    ]
    save_jsonl(output, f"{OUTPUT_PATH_SFT}/{role}_reaction.jsonl")


def convert_self_knowledge(role):
    data = load_json(f"{DATASET_PATH}/self_knowledge.json")
    output = [
        {
            "instruction": f"You are playing the role of {role}, answer the question as {role}.",
            "input": entry["instruction"],
            "output": "True" if entry["answer"] else "False"
        }
        for entry in data if entry["target_role"] == role
    ]
    save_jsonl(output, f"{OUTPUT_PATH_SFT}/{role}_self_knowledge.jsonl")


def convert_summary(role):
    data = load_json(f"{DATASET_PATH}/summary.json")
    output = [
        {
            "instruction": f"You are playing the role of {role}, answer the question as {role}.",
            "input": f"{entry['source_role']}：{entry['question']}",
            "output": entry["summary"]
        }
        for entry in data if entry["target_role"] == role
    ]
    save_jsonl(output, f"{OUTPUT_PATH_SFT}/{role}_summary.jsonl")


# ---- DPO 数据转换 ----


def convert_to_conversation_chosen(role, conversation_path, chosen_path):
    conversations = load_conversation(conversation_path)
    chosen_data = []
    grouped_by_scene = {}

    for entry in conversations:
        scene_id = entry["scene_id"]
        grouped_by_scene.setdefault(scene_id, []).append(entry)

    for scene_id, scene_conversations in grouped_by_scene.items():
        history = []
        for i, entry in enumerate(tqdm(scene_conversations, desc=f"Processing scene {scene_id}")):
            if role in entry["role"].split(","):
                input_data = "\n".join([
                    f'"{turn["role"]}": "{turn["content"]}"'
                    for turn in scene_conversations[:i]
                    if not any(f'"{turn["role"]}": "{turn["content"]}"' in h[0] or f'"{turn["role"]}": "{turn["content"]}"' in h[1] for h in history)
                ])
                if input_data:
                    chosen_data.append({
                        "instruction": f"You are playing the role of {role}. Please fully immerse yourself in the identity of {role} and respond accordingly.\n\n"
                                       f"Do not break character or provide out-of-character (OOC) explanations. Focus solely on responding as {role}. Provide the next line of dialogue in {role}'s tone.",
                        "input": input_data,
                        "chosen": entry["content"],
                        "history": history.copy()
                    })
                    history.append([input_data, f'"{role}": "{entry["content"]}"'])

    save_jsonl(chosen_data, chosen_path)
    print(f"Saved conversation_chosen to {chosen_path}")


def fill_in_dpo_generation_template(role, input_data, chosen, broken_style):
    return f'''Historical dialogue: {input_data}, Correct answer: {chosen}.

Based on the correct answer, provide an alternative response (rejected) that is significantly different from {role}'s tone.
The incorrect response should be in the {broken_style} style but maintain the same meaning as the original sentence.
Avoid any analysis or phrases like "Here is the answer:".

Example output format:

- rejected: Response
'''



def convert_to_conversation_full(role, chosen_path, full_path, model_engine, token):
    if os.path.exists(full_path):
        print(f"{full_path} 已存在，跳过生成步骤。")
        return

    chosen_data = load_conversation(chosen_path)
    full_data = []
    broken_styles = ["formal", "classical", "translated tone", "overly emotional", "opposite gender tone"]

    for item in tqdm(chosen_data, desc="Generating full conversations"):
        broken_style = random.choice(broken_styles)
        messages = [
            {"role": "system", "content": f"You are playing the role of {role}. Please fully immerse yourself in the identity of {role} and respond accordingly."},
            {"role": "user", "content": fill_in_dpo_generation_template(role, item['input'], item['chosen'], broken_style)}
        ]
        _, rejected = call_tsinghua_deepseek(model_engine, token, messages)
        print("messages: ", messages)
        print("rejected: ", rejected)

        full_data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "chosen": item["chosen"],
            "rejected": rejected.replace('- rejected ', '').strip(),
            "history": item["history"]
        })

    save_jsonl(full_data, full_path)
    print(f"Saved conversation_full to {full_path}")


# ---------------------- Wiki 数据转换 ----------------------

def load_wiki_data(role):
    wiki_file = f"{WIKI_PATH}/wiki_{role}.txt"
    with open(wiki_file, 'r', encoding='utf-8') as f:
        return [p.strip() for p in f.read().split('\n\n') if p.strip()]

def fill_in_convert_to_statement_template(character, passage):
    return f'''Given the following passage about "{character}":

{passage}

Please generate some important character-setting statements about "{character}" for role-playing AI to follow.

- No matter the passage is in Chinese or English, you should generate statements in English.
- Strictly follow the format below, with each statement starting with "- ".
- Ensure each statement explicitly mentions "{character}" and avoids pronouns or co-references.
- Retain as much information as possible from the passage, especially details like book titles, locations, birthdays, or organizations.
- State facts and avoid vague phrases like "patterns," "traits," or "paradigms."
- Focus only on the given passage and do not reference historical conversations.
- Avoid introductory phrases like "Here is the answer:".

Example output format:

- {character} is...
- {character} has...
- {character} primarily studies...
- {character}'s works include...

Strictly follow these instructions when generating statements.
'''

def generate_statements(role, model_engine, token):
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{role}_statement.json"
    if os.path.exists(statement_path):
        print(f"{statement_path} already exists. Skipping generation.")
        return

    os.makedirs(os.path.dirname(statement_path), exist_ok=True)
    wiki_data = load_wiki_data(role)
    results = []

    for passage in tqdm(wiki_data, desc="Generating Statements"):
        messages = [
            {"role": "system", "content": "You are a language rewriting assistant helping users construct statements from passages."},
            {"role": "user", "content": fill_in_convert_to_statement_template(role, passage)}
        ]
        _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
        print("messages:", messages)
        print("generated_content: ", generated_content)
        statements = [s.strip() for s in generated_content.split('\n') if s.startswith('- ')]
        results.append({"passage": passage, "statements": statements})

    save_json(results, statement_path)

def fill_in_relevant_query_generation_template(character, statement):
    return f'''Character statement: {statement}

You need to ask some questions about {character} that require responses based on the information in the above statement.

Provide 3 diverse and concise possible questions, treating the conversation partner as {character} without mentioning their name. Strictly follow the example format below, avoid unnecessary analysis, and do not include phrases like "Here is the answer:".

Example output format:

- Are you...?
- How do you view...?
'''

def generate_queries(role, model_engine, token):
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{role}_statement.json"
    with_query_path = f"{OUTPUT_PATH_WITH_QUERY}/{role}_with_query.json"

    if os.path.exists(with_query_path):
        print(f"{with_query_path} already exists. Skipping generation.")
        return

    os.makedirs(os.path.dirname(with_query_path), exist_ok=True)
    statements_data = load_json(statement_path)
    results = []

    for item in tqdm(statements_data, desc="Generating Queries"):
        for statement in item["statements"]:
            messages = [
                {"role": "system", "content": "You are a language rewriting assistant helping users construct questions from statements."},
                {"role": "user", "content": fill_in_relevant_query_generation_template(role, statement)}
            ]
            _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
            print("messages:", messages)
            print("generated_content: ", generated_content)
            queries = [q.strip() for q in generated_content.split('\n') if q.startswith('- ')]
            results.append({"statement": statement, "queries": queries})

    save_json(results, with_query_path)

def fill_in_rejected_template(role, statement, query):
    return f'''Character statement: {statement}, Question: {query},

Based on the statement and question, respond as {role}. Provide a correct answer (chosen) and an incorrect answer (rejected).
The correct answer must fully align with the information in the statement and question, and be phrased in {role}'s tone. The incorrect answer must significantly contradict some part of the statement. Strictly follow the example format below, avoid unnecessary analysis, and do not include phrases like "Here is the answer:".

Example output format:

- chosen: Answer 1
- rejected: Answer 2
'''

def generate_rejected(role, model_engine, token):
    with_query_path = f"{OUTPUT_PATH_WITH_QUERY}/{role}_with_query.json"
    dpo_path = f"{OUTPUT_PATH_DPO}/{role}_generated_qa.json"

    if os.path.exists(dpo_path):
        print(f"{dpo_path} already exists. Skipping generation.")
        return

    os.makedirs(os.path.dirname(dpo_path), exist_ok=True)
    with_query_data = load_json(with_query_path)
    results = []

    for item in tqdm(with_query_data, desc="Generating DPO"):
        for query in item["queries"]:
            instruction = f"You are playing the role of {role}. Based on the statement '{item['statement']}', answer the question as {role}.\n"
            messages = [
                {"role": "system", "content": "You are a language rewriting assistant helping users construct datasets."},
                {"role": "user", "content": fill_in_rejected_template(role, item['statement'], query)}
            ]
            _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
            print("messages:", messages)
            print("generated_content: ", generated_content)
            answers = [a.strip() for a in generated_content.split('\n') if a.startswith('- ')]
            if len(answers) >= 2:
                results.append({
                    "instruction": instruction,
                    "input": query.replace('- ', ''),
                    "chosen": answers[0].replace('- chosen: ', ''),
                    "rejected": answers[1].replace('- rejected: ', '')
                })

    save_json(results, dpo_path)

def shuffle_and_save(input_pattern, output_file):
    # 匹配输入文件
    input_files = glob.glob(input_pattern)
    all_data = []

    # 读取所有文件的数据
    for file in input_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # 检查是否有 history 字段，如果没有则补齐为空列表
                if "history" not in item:
                    item["history"] = []
                all_data.append(item)

    # 打乱数据顺序
    random.shuffle(all_data)

    # 保存为 JSON 数组
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"Saved shuffled data to {output_file}")

def main(role, model_engine, token):
    # SFT 转换
    convert_general_response(role)
    convert_reaction(role)
    convert_self_knowledge(role)
    convert_summary(role)

    # DPO 转换
    conversation_path = f"{DATASET_PATH}/profiles/{role}.jsonl"
    chosen_path = f"{OUTPUT_PATH_CHOSEN}/{role}_chosen.jsonl"
    full_path = f"{OUTPUT_PATH_DPO}/{role}_conversation.jsonl"

    convert_to_conversation_chosen(role, conversation_path, chosen_path)
    convert_to_conversation_full(role, chosen_path, full_path, model_engine, token) # 这里会调用 DeepSeek 模型
    
    generate_statements(role, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_queries(role, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_rejected(role, model_engine, token) # 这里会调用 DeepSeek 模型
    

    # 处理 sft 数据
    shuffle_and_save(f"{OUTPUT_PATH_SFT}/{role}_*.jsonl", f"{OUTPUT_PATH_BASE}/{role}_sft_shuffle.json")

    # 处理 dpo 数据
    shuffle_and_save(f"{OUTPUT_PATH_DPO}/{role}_*.jsonl", f"{OUTPUT_PATH_BASE}/{role}_dpo_shuffle.json")


# 示例调用
main("刘梅", "DeepSeek-R1-Distill-32B", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0Mjk4OTA4MSwiZXhwIjoxNzQzMDEwNjgxfQ.sdgkd8Liy5v8VPzaz5jyz2L-bLvFBXD4lpVXukrS11U")