import glob
import os
import json
import random
import time
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

#改到v2之后新增的输出，有用的东西是f"{OUTPUT_PATH_BASE}/cot/{role}_cot.jsonl"，和f"{OUTPUT_PATH_BASE}/style/{role}_style_transfer.jsonl"
OUTPUT_PATH_COT = f"{OUTPUT_PATH_BASE}/cot" # 一个输出文件，cot
OUTPUT_PATH_STYLE = f"{OUTPUT_PATH_BASE}/style" # 一个输出文件，style


def call_tsinghua_deepseek(model, token, messages, max_retries=5, base_delay=1, max_delay=16):
    """
    调用清华大学的 DeepSeek 模型，支持指数退避重试逻辑
    :param model: 模型名称，例如 "DeepSeek-R1-671B"
    :param token: 授权令牌
    :param messages: 请求的消息内容
    :param max_retries: 最大重试次数
    :param base_delay: 初始延迟时间（秒）
    :param max_delay: 最大延迟时间（秒）
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

    attempt = 0
    while attempt < max_retries:
        try:
            buffer = ""  # 缓冲池
            full_content = ""  # 用于累积所有内容

            with requests.post(url, headers=headers, json=data, stream=True, timeout=30) as r:
                # print(f"请求状态码: {r.status_code}")
                if r.status_code != 200:
                    print(f"请求失败，响应内容: {r.text}")
                    raise requests.exceptions.RequestException(f"HTTP {r.status_code}")

                # 逐行读取响应流
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        # print(f"接收到的行: {decoded_line}")  # 打印每一行返回内容
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

            print(f"完整返回内容: {full_content}")  # 打印完整的返回内容

            # 在最后进行分割
            if "</think>" in full_content:
                think, response = full_content.split("</think>")
                think, response = think.strip(), response.strip()
            else:
                think, response = "", full_content.strip()

            # 如果 think 和 response 都为空，触发重试
            if not think and not response:
                raise ValueError("输出为空，触发重试")

            return think, response

        except (requests.exceptions.RequestException, ValueError) as e:
            attempt += 1
            delay = min(base_delay * (2 ** attempt), max_delay)  # 指数退避计算延迟时间
            print(f"请求异常: {e}. 第 {attempt} 次重试将在 {delay:.2f} 秒后进行...")
            time.sleep(delay + random.uniform(0, 0.5))  # 增加随机抖动以避免请求冲突

    print("请求失败，已达到最大重试次数。")
    return "", ""


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


def cot_history_input():
    """
    history的input
    """
    return f'''根据以下内容生成思考过程，严格遵守示例输出格式，为我提供推理过程：
- 输入 "你正在扮演 刘星，请完全沉浸在该角色的身份中进行回复。\n\n不要跳出角色或提供 OOC（Out of Character）的解释，仅专注于 刘星 的身份进行对话。请以 刘星 的口吻给出回答。你的调皮劲是不是遗传自妈妈？"
- 输出 "那必须的！我妈年轻时候比我还能闹腾呢，我这叫青出于蓝！"

示例输出格式：

### 推理过程：
1. 【问题重述】
让我们一步一步思考。这段对话中，用户向扮演xx的角色提问，问题是：“……”

2. 【确定背景】
人物确认：问题中的“你”是xx，“xx”指的是……。
背景补充：故事发生在《xxx》（电视剧/电影/小说/历史），其中xxx是……。
诱导性判别：问题是否有事实性错误，扮演的xx是否能够回答问题
情境匹配：问题描述的是xx时，xx的反应，这与陈述“xx”相关/不相关，因此（是否）需要结合陈述进行回答。

3. 【推理与选择】
问题能否被扮演的角色回答；问题（是否）需要结合陈述来构建回答。

合理的回答应体现：
a. xx具有xx的性格；
b. xx曾经xx；
c. 回答应符合xx状态的xx语气。
'''


def cot_history_output():
    """
    history的output
    """
    return f'''1. 【问题重述】
让我们一步一步思考。这段对话中，用户向扮演“刘星”的角色提问，问题是：“你的调皮劲是不是遗传自妈妈？”

2. 【确定背景】

人物确认：问题中的“你”是“刘星”。

“刘星”指的是：《家有儿女》中的主要角色之一，一个调皮捣蛋但聪明幽默的小男孩。

背景补充：《家有儿女》是一部中国家庭情景喜剧，讲述了重组家庭中三个孩子与父母之间的日常生活趣事。刘星是其中的“中间儿子”，极具表现力、活泼好动。

诱导性判别：问题“你的调皮劲是不是遗传自妈妈？”属于对角色性格来源的轻松调侃，不涉及事实性错误，也不需要脱离角色解释。该问题可以通过角色设定进行回答。

情境匹配：该问题假设“调皮”是刘星的显著特点（符合角色设定），同时提到“妈妈”，也正好呼应剧中刘星与母亲刘梅之间的互动。因此需要结合角色设定和剧中家庭关系来构建回答。

3. 【推理与选择】

问题能否被扮演的角色回答：可以。刘星在剧中常用夸张幽默的方式描述父母，且经常夸口、爱“斗嘴”。

是否需要结合陈述来构建回答：是。角色设定和语气应体现刘星的性格特征。

合理的回答应体现：
a. 刘星具有调皮、嘴贫、聪明的性格特点；
b. 刘星经常与父母“抬杠”，尤其是与妈妈刘梅斗嘴；
c. 语气应轻松、俏皮、略带夸张，符合喜剧角色特质。

4. 【回答】

- 动作：一脸得意地咧嘴笑，眼睛一眯，明显在“显摆”。

- 情绪：自豪中带点调侃，语气上扬。

- 思绪：心里想着：“这个问题问得好，我正好可以调侃一下老妈。”

- 语气：夸张俏皮、口语化，像在跟朋友炫耀。

- 输出："那必须的！我妈年轻时候比我还能闹腾呢，我这叫青出于蓝！"
'''

def fill_in_cot_template(role, input_data, chosen):
    """
    填充生成思考过程的模板，instruction被替换为固定的instruction
    """
    return f'''根据以下内容生成思考过程，严格遵守示例输出格式，为我提供推理过程以及输出：
- 指令：你正在扮演 {role}，请完全沉浸在该角色的身份中进行回复。\n\n不要跳出角色或提供 OOC（Out of Character）的解释，仅专注于{role} 的身份进行对话。请以 {role} 的口吻给出回答。
- 输入：{input_data}
- 输出：{chosen}

### 推理过程：
1. 【问题重述】
让我们一步一步思考。这段对话中，用户向扮演{role}的角色提问，问题是：“……”

2. 【确定背景】
人物确认：问题中的“你”是xx，“xx”指的是……。
背景补充：故事发生在《xxx》（电视剧/电影/小说/历史），其中xxx是……。
诱导性判别：问题是否有事实性错误，扮演的{role}是否能够回答问题
情境匹配：问题描述的是xx时，xx的反应，这与陈述“xx”相关/不相关，因此（是否）需要结合陈述进行回答。

3. 【推理与选择】
问题能否被扮演的角色回答；问题（是否）需要结合陈述来构建回答。

合理的回答应体现：
a. {role}具有xx的性格；
b. xx曾经xx；
c. 回答应符合xx状态的{role}语气。

4. 【回答】
- 动作：
- 情绪：
- 思绪：
- 语气：
- 回答：
'''

def generate_cot(role, model_engine, token):
    """
    从 DPO 数据生成 CoT（思考过程）
    """
    dpo_path = f"{OUTPUT_PATH_DPO}/{role}_generated_qa.jsonl"
    cot_path = f"{OUTPUT_PATH_COT}/{role}_cot.jsonl"

    # 检查文件是否存在
    if os.path.exists(cot_path):
        print(f"{cot_path} 已存在，跳过生成步骤。")
        return

    # 创建目标目录
    os.makedirs(os.path.dirname(cot_path), exist_ok=True)

    # 读取 DPO 数据
    dpo_data = load_conversation(dpo_path)
    cot_data = []

    for item in tqdm(dpo_data, desc="Generating CoT"):
        # 填充模板
        cot_prompt = fill_in_cot_template(role, item["input"], item["chosen"])
        messages = [
            {"role": "user", "content": cot_history_input()},
            {"role": "assistant", "content": cot_history_output()},
            {"role": "user", "content": cot_prompt}
        ]

        # 调用 DeepSeek 模型生成思考过程
        _, generated_cot = call_tsinghua_deepseek(model_engine, token, messages)
        print("messages:", messages)
        print("*******************")
        print("generated_cot:", generated_cot)
        
        # 检查是否有 history 字段，如果没有则补齐为空列表
        if "history" not in item:
            item["history"] = []

        # 保存生成的 CoT 数据
        cot_data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "cot": generated_cot.strip(),
            "history": item["history"]
        })

    # 保存为 JSONL 文件
    save_jsonl(cot_data, cot_path)
    print(f"Saved CoT data to {cot_path}")

def generate_style_transfer(role):
    """
    从 conversation 数据生成风格迁移数据
    """
    conversation_path = f"{OUTPUT_PATH_DPO}/{role}_conversation.jsonl"
    style_transfer_path = f"{OUTPUT_PATH_STYLE}/{role}_style_transfer.jsonl"

    # 检查文件是否存在
    if os.path.exists(style_transfer_path):
        print(f"{style_transfer_path} 已存在，跳过生成步骤。")
        return

    # 创建目标目录
    os.makedirs(os.path.dirname(style_transfer_path), exist_ok=True)

    # 读取 conversation 数据
    conversation_data = load_conversation(conversation_path)
    style_transfer_data = []

    for item in tqdm(conversation_data, desc="Generating Style Transfer"):
        # 构造新的 JSON 数据
        style_transfer_data.append({
            "instruction": f"你正在扮演 {role}，你需要将下面的句子转写成 {role} 的口吻",
            "input": item["rejected"],
            "output": item["chosen"],
            "history": item["history"]
        })

    # 保存为 JSONL 文件
    save_jsonl(style_transfer_data, style_transfer_path)
    print(f"Saved Style Transfer data to {style_transfer_path}")

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
    # modified feice Apr 15, 2025 at 22:10 
    # 我读了下RoleAgent（数据集来源），他是把这里的数据用来当评估的。所以现在不用这些训了。
    # convert_general_response(role)
    # convert_reaction(role)
    # convert_self_knowledge(role)
    # convert_summary(role)

    # DPO 转换
    conversation_path = f"{DATASET_PATH}/profiles/{role}.jsonl"
    chosen_path = f"{OUTPUT_PATH_CHOSEN}/{role}_chosen.jsonl"
    full_path = f"{OUTPUT_PATH_DPO}/{role}_conversation.jsonl"

    convert_to_conversation_chosen(role, conversation_path, chosen_path)
    convert_to_conversation_full(role, chosen_path, full_path, model_engine, token) # 这里会调用 DeepSeek 模型
    
    generate_statements(role, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_queries(role, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_rejected(role, model_engine, token) # 这里会调用 DeepSeek 模型

    # 生成CoT数据
    generate_cot(role, model_engine, token) # 这里会调用 DeepSeek 模型
    
    # 生成Style数据
    generate_style_transfer(role)

    # 处理 sft 数据
    # modified feice Apr 15, 2025 at 22:11
    # shuffle_and_save(f"{OUTPUT_PATH_SFT}/{role}_*.jsonl", f"{OUTPUT_PATH_BASE}/{role}_sft_shuffle.json")

    # 处理 dpo 数据
    # shuffle_and_save(f"{OUTPUT_PATH_DPO}/{role}_*.jsonl", f"{OUTPUT_PATH_BASE}/{role}_dpo_shuffle.json")


# 示例调用
main("刘星", "DeepSeek-R1-Distill-32B", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0NDcyMjYxMywiZXhwIjoxNzQ0NzQ0MjEzfQ.fTlWKjlF1Q6Twa28qbTv0TKknK4IATS7OGlMrvZnuQw")
main("刘梅", "DeepSeek-R1-Distill-32B", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0NDcyMjYxMywiZXhwIjoxNzQ0NzQ0MjEzfQ.fTlWKjlF1Q6Twa28qbTv0TKknK4IATS7OGlMrvZnuQw")
main("夏东海", "DeepSeek-R1-Distill-32B", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0NDcyMjYxMywiZXhwIjoxNzQ0NzQ0MjEzfQ.fTlWKjlF1Q6Twa28qbTv0TKknK4IATS7OGlMrvZnuQw")
main("小雨", "DeepSeek-R1-Distill-32B", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0NDcyMjYxMywiZXhwIjoxNzQ0NzQ0MjEzfQ.fTlWKjlF1Q6Twa28qbTv0TKknK4IATS7OGlMrvZnuQw")
main("小雪", "DeepSeek-R1-Distill-32B", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0NDcyMjYxMywiZXhwIjoxNzQ0NzQ0MjEzfQ.fTlWKjlF1Q6Twa28qbTv0TKknK4IATS7OGlMrvZnuQw")
