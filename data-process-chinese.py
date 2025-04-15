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
            "instruction": f"你正在扮演{role}，请以{role}的身份回答问题",
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
            "instruction": f"你正在扮演{role}，请以{role}的身份，回答下面的问题。",
            "input": f"{entry['source_role']}：{entry['question']}，在下面的选项中选择答案，直接回答编号：{entry['multi_choices']}",
            "output": entry["gt_answer"]
        }
        for entry in data if entry["target_role"] == role
    ]
    save_jsonl(output, f"{OUTPUT_PATH_SFT}/{role}_reaction.jsonl")


def convert_self_knowledge(role):
    data = load_json(f"{DATASET_PATH}/self_knowledge.json")
    output = [
        {
            "instruction": f"你正在扮演{role}，请以{role}的身份，回答下面的问题。",
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
            "instruction": f"你正在扮演{role}，请以{role}的身份，回答下面的问题。",
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
                        "instruction": f"你正在扮演 {role}，请完全沉浸在该角色的身份中进行回复。\n\n"
                                       f"不要跳出角色或提供 OOC（Out of Character）的解释，仅专注于 {role} 的身份进行对话。请以 {role} 的口吻给出下一句对话",
                        "input": input_data,
                        "chosen": entry["content"],
                        "history": history.copy()
                    })
                    history.append([input_data, f'"{role}": "{entry["content"]}"'])

    save_jsonl(chosen_data, chosen_path)
    print(f"Saved conversation_chosen to {chosen_path}")


def fill_in_dpo_generation_template(role, input_data, chosen, broken_style):
    return f'''历史对话：{input_data}，正确回答：{chosen}。

参考正确回答，以明显不同于{role}的口吻，为我提供一个另外的回答（rejected）。
这个错误的回答应为输入的{broken_style}版本, 但语句的含义与原句保持一致。
不需要任何分析，避免诸如"以下是答案："之类的陈述。

示例输出格式：

- rejected: 回答
'''



def convert_to_conversation_full(role, chosen_path, full_path, model_engine, token):
    if os.path.exists(full_path):
        print(f"{full_path} 已存在，跳过生成步骤。")
        return

    chosen_data = load_conversation(chosen_path)
    full_data = []
    broken_styles = ["书面语", "古文", "翻译腔", "过度情绪化", "异性口吻"]

    for item in tqdm(chosen_data, desc="Generating full conversations"):
        broken_style = random.choice(broken_styles)
        messages = [
            {"role": "system", "content": f"你正在扮演 {role}，请完全沉浸在该角色的身份中进行回复。"},
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
    return f'''给定关于"{character}"的段落：

{passage}

请生成一些关于"{character}"的重要人设陈述，供角色扮演的AI遵循。

- 严格遵循以下格式，每个陈述以"- "开头。
- 确保每个陈述中明确提到"{character}"，避免使用代词或共指。
- 尽可能保留段落中的信息，特别是书名、地点、生日或组织等实体细节。
- 指出事实，而不要给出空洞的话语，避免使用"模式"、"特性"、"范式"类型的词语。
- 仅关注给定的段落，不要引用历史对话中的信息。
- 避免生成诸如"以下是答案："之类的介绍性短语。

示例输出格式：

- {character}是...
- {character}有...
- {character}主要研究...
- {character}的作品包括...

生成陈述时严格遵循这些指示。
'''

def generate_statements(role, model_engine, token):
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
            {"role": "user", "content": fill_in_convert_to_statement_template(role, passage)}
        ]
        _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
        print("messages:", messages)
        print("generated_content: ", generated_content)
        statements = [s.strip() for s in generated_content.split('\n') if s.startswith('- ')]
        results.append({"passage": passage, "statements": statements})

    with open(statement_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    # save_jsonl(results, statement_path)

def fill_in_relevant_query_generation_template(character, statement):
    return f'''人设陈述：{statement}

你需要对{character}提出一些问题，这些问题需要包含上述人设陈述中的信息进行回应。

提供3个多样且简洁的可能话语，这些话语将谈话对象视为{character}，且不包含名字。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- 你是……吗？
- 你如何看待……'''

def generate_queries(role, model_engine, token):
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{role}_statement.json"
    with_query_path = f"{OUTPUT_PATH_WITH_QUERY}/{role}_with_query.json"

    if os.path.exists(with_query_path):
        print(f"{with_query_path} 已存在，跳过生成步骤。")
        return

    os.makedirs(os.path.dirname(with_query_path), exist_ok=True)
    statements_data = load_json(statement_path)
    results = []

    for item in tqdm(statements_data, desc="Generating Queries"):
        for statement in item["statements"]:
            messages = [
                {"role": "system", "content": "你是一个语言改写助手，帮助用户从陈述构筑问题。"},
                {"role": "user", "content": fill_in_relevant_query_generation_template(role, statement)}
            ]
            _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
            print("messages:", messages)
            print("generated_content: ", generated_content)
            queries = [q.strip() for q in generated_content.split('\n') if q.startswith('- ')]
            results.append({"statement": statement, "queries": queries})

    with open(with_query_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def fill_in_rejected_template(role, statement, query):
    return f'''人设陈述：{statement}，问题：{query}，
    
根据陈述和问题，以{role}的身份回答问题。为我提供一个正确的回答（chosen）和错误的回答（rejected）。
正确的回答需要完全符合陈述和问题的信息，并以{role}的口吻说出。错误的回答则需要与陈述的某处有重要不一致。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- chosen: 回答1
- rejected: 回答2
'''

def generate_rejected(role, model_engine, token):
    """从 with_query 生成 dpo 并存储。"""
    with_query_path = f"{OUTPUT_PATH_WITH_QUERY}/{role}_with_query.json"
    dpo_path = f"{OUTPUT_PATH_DPO}/{role}_generated_qa.jsonl"
    
    # 检查文件是否存在
    if os.path.exists(dpo_path):
        print(f"{dpo_path} 已存在，跳过生成步骤。")
        return

    # 创建目标目录
    os.makedirs(os.path.dirname(dpo_path), exist_ok=True)
    
    with open(with_query_path, 'r', encoding='utf-8') as f:
        with_query_data = json.load(f)
    
    results = []
    for item in tqdm(with_query_data, desc="Generating DPO"):
        for query in item["queries"]:
            instruction = f"你正在扮演{role}，关于{role}有陈述“{item['statement']}”，请以{role}的身份回答问题\n"
            messages = [
                {"role": "system", "content": "你是一个语言改写助手，帮助用户构筑数据集。"},
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
                    "rejected": answers[1].replace('- rejected ', '')
                })
    
    # with open(dpo_path, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)
    save_jsonl(results, dpo_path)

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
main("刘梅", "DeepSeek-R1-671B", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0Mjk4OTA4MSwiZXhwIjoxNzQzMDEwNjgxfQ.sdgkd8Liy5v8VPzaz5jyz2L-bLvFBXD4lpVXukrS11U")
main("小雪", "DeepSeek-R1-671B", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0Mjk4OTA4MSwiZXhwIjoxNzQzMDEwNjgxfQ.sdgkd8Liy5v8VPzaz5jyz2L-bLvFBXD4lpVXukrS11U")
main("小雨", "DeepSeek-R1-671B", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0Mjk4OTA4MSwiZXhwIjoxNzQzMDEwNjgxfQ.sdgkd8Liy5v8VPzaz5jyz2L-bLvFBXD4lpVXukrS11U")
