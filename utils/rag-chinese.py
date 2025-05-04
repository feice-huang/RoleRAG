"""
# modified feice Apr 28, 2025 at 16:40
1. 用于生成RAG需要的SFT数据，数据格式与SmartRAG类似
    a. Answer Directly: [Answer] xxx
    b. Retrieve(Rewrite): [Retrieve] xxx
    c. Answer with Observations: [Answer] xxx
2. 先去看看SmartRAG有没有开源数据，能不能抄进来
    a. 很遗憾，没有开源数据集构建的步骤
    b. 他这个数据集也太大了，JSON 数组中共有 105864 条数据，人工真能删的过来吗
3. 重新想一下这个方法，优势在于能够在对话历史中做检索，能够增强记忆。
3. 总之先大规模生成吧，问题沿用CoT的
    a. 考虑直接从deepseek中蒸馏能力，比如问题重写，可以直接让deepseek按要求生成
    b. 可以，那所有的能力全都从deepseek中蒸馏出来
"""

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
WIKI_PATH = "/data/hfc/mydata/input/wiki"
ANTI_PATH = "/data/hfc/mydata/input/anti" # 反例关键词的路径
GENERAL_PATH = "/data/hfc/mydata/input/general" # 防止生成数据集时引入幻觉，每次都要传给deepseek general信息
SEED_PATH = "/data/hfc/RoleRAG/mydata/input/seed"

# 输出数据的路径，最后有用的东西是f"{OUTPUT_PATH_BASE}/{role}_sft_shuffle.jsonl"和f"{OUTPUT_PATH_BASE}/{role}_dpo_shuffle.jsonl"，分别是sft和dpo的数据
OUTPUT_PATH_BASE = "/data/hfc/mydata"
PROCESS_PATH_BASE = "/data/hfc/mydata/process"

# 输出数据的路径，用于存储中间数据
OUTPUT_PATH_SFT = f"{PROCESS_PATH_BASE}/sft" # 四个中间文件，general_response, reaction, self_knowledge, summary
OUTPUT_PATH_DPO = f"{PROCESS_PATH_BASE}/dpo" # 两个中间文件，conversation和generated_qa
OUTPUT_PATH_STATEMENT = f"{PROCESS_PATH_BASE}/statement" # 一个中间文件，statement
OUTPUT_PATH_WITH_QUERY = f"{PROCESS_PATH_BASE}/with_query" # 一个中间文件，with_query
OUTPUT_PATH_CHOSEN = f"{PROCESS_PATH_BASE}/chosen" # 一个中间文件，chosen

# 改到v2之后新增的输出，有用的东西是f"{OUTPUT_PATH_BASE}/cot/{role}_cot.jsonl"，和f"{OUTPUT_PATH_BASE}/style/{role}_style.jsonl"
OUTPUT_PATH_COT = f"{PROCESS_PATH_BASE}/cot" # 两个输出文件，cot、noise
OUTPUT_PATH_STYLE = f"{PROCESS_PATH_BASE}/style" # 一个输出文件，style

# 改到v3后新增的输出
OUTPUT_PATH_RECALL = f"{PROCESS_PATH_BASE}/recall" # 一个中间文件，recall

# 改到RAG后新增的输出
OUTPUT_PATH_RAG = f"{PROCESS_PATH_BASE}/rag" # 一个中间文件，rag


def call_tsinghua_deepseek(model, token, messages, max_retries=5, base_delay=1, max_delay=16):
    """
    调用清华大学的 DeepSeek 模型，支持指数退避重试逻辑
    :param model: 模型名称，例如 "DeepSeek-R1-671B" 或 "DeepSeek-R1-Distill-32B"
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

def fill_in_world_statement_template(world, passage):
    return f'''给定关于"{world}"的段落：

{passage}

请生成一些关于"{world}"的重要陈述。

- 严格遵循以下格式，每个陈述以"- "开头。
- 确保每个陈述中明确提到"{world}"，避免使用代词或共指。
- 尽可能保留段落中的信息，特别是书名、地点、日期或组织等实体细节，不要捏造任何没有提到的事实。
- 指出事实，而不要给出空洞的话语，避免使用"模式"、"特性"、"范式"类型的词语。
- 仅关注给定的段落，不要引用历史对话中的信息。
- 用尽量少的陈述来覆盖段落中的所有信息，同一信息不要多次出现。
- 避免生成诸如"以下是答案："之类的介绍性短语。

示例输出格式：

- {world}是什么样的作品...
- {world}中的情节：...
- {world}中的人物：...

生成陈述时严格遵循这些指示。
'''

def generate_world_statements(world, model_engine, token):
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{world}_statement.json"
    if os.path.exists(statement_path):
        print(f"{statement_path} 已存在，跳过生成步骤。")
        return

    os.makedirs(os.path.dirname(statement_path), exist_ok=True)
    wiki_data = load_wiki_data(world)
    results = []

    for passage in tqdm(wiki_data, desc="Generating World Statements"):
        messages = [
            {"role": "system", "content": "你是一个语言改写助手，帮助用户从段落构筑陈述。"},
            {"role": "user", "content": fill_in_world_statement_template(world, passage)}
        ]
        _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
        print("messages:", messages)
        print("generated_content: ", generated_content)
        statements = [s.strip() for s in generated_content.split('\n') if s.startswith('- ')]
        results.append({"passage": passage, "statements": statements})

    with open(statement_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def fill_in_relevant_query_role_template(character, statement, general):
    return f'''已知关于{character}的背景信息：

{general}

人设陈述：{statement}

你需要对{character}提出一些问题，这些问题需要包含上述人设陈述中的信息进行回应。不需要涉及背景信息，保持不违背即可。

提供3个多样且简洁的可能话语，这些话语将谈话对象视为{character}，且不包含名字。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- 你是{character}吗
- 你对你的这一段经历有什么看法'''

def generate_role_queries(role, model_engine, token):
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{role}_statement.json"
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    with_query_path = f"{OUTPUT_PATH_WITH_QUERY}/{role}_with_query.json"

    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()
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
                {"role": "user", "content": fill_in_relevant_query_role_template(role, statement, general)}
            ]
            _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
            print("messages:", messages)
            print("generated_content: ", generated_content)
            queries = [q.strip() for q in generated_content.split('\n') if q.startswith('- ')]
            results.append({"statement": statement, "queries": queries})

    with open(with_query_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def fill_in_role_anti_template(world, role, question_type, description, keyword, general):
    return f'''已知关于{role}的背景信息：

{general}

你正在对AI扮演的{world}中的{role}进行诱导性提问。

请问出一个{role}不应该回答上来的问题，并给出一个表现出恰当迷茫和不解的回答。
注意，当问题中仅有一部分超出认知时，需要给出一个模糊的回答，而不是完全不知所云。

问题类型：{question_type}
问题描述：{description}
问题关键词：{keyword}

严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- query: 你了解ChatGPT吗？
- answer: 那是什么东西？学校的英语课没学过这个单词啊。'''


def generate_role_anti_queries(world, role, model_engine, token):
    """
    生成针对角色的诱导性提问（反例问题）。
    :param world: 世界名称
    :param role: 角色名称
    :param model_engine: 模型名称
    :param token: 授权令牌
    """
    anti_path = os.path.join(ANTI_PATH, f"anti_{role}.json")
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    with_query_path = os.path.join(OUTPUT_PATH_WITH_QUERY, f"{role}_with_anti_query.json")

    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()
        
    # 检查文件是否存在
    if os.path.exists(with_query_path):
        print(f"{with_query_path} 已存在，跳过生成步骤。")
        return

    # 创建目标目录
    os.makedirs(os.path.dirname(with_query_path), exist_ok=True)

    # 读取反例关键词数据
    anti_data = load_json(anti_path)
    results = []

    for item in tqdm(anti_data, desc=f"Generating Anti Queries for {role}"):
        question_type = item["type"]
        description = item["description"]
        example_keywords = item["example_keywords"]

        for keyword in example_keywords:
            # 调用模型生成问题和回答
            messages = [
                {"role": "system", "content": "你是一个语言改写助手，帮助用户生成诱导性问题和回答。"},
                {"role": "user", "content": fill_in_role_anti_template(world, role, question_type, description, keyword, general)}
            ]
            _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
            print("messages:", messages)
            print("generated_content:", generated_content)

            # 解析生成的内容
            queries_and_answers = [
                qa.strip() for qa in generated_content.split('\n') if qa.startswith('- ')
            ]
            query, answer = None, None
            for qa in queries_and_answers:
                if qa.startswith('- query:'):
                    query = qa.replace('- query: ', '').strip()
                elif qa.startswith('- answer:'):
                    answer = qa.replace('- answer: ', '').strip()
            if query and answer:
                results.append({"query": query, "answer": answer})

    # 保存生成的反例问题和回答
    with open(with_query_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Saved anti queries to {with_query_path}")

def fill_in_relevant_query_role_instruction_template(character):
    return f'''下面是一段关于{character}的问题，请为我提供用于回复这些问题的信息。

按问题涉及的信息不同，提供数个多样且简洁的可能信息。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- {character}是
- {character}看待'''

def generate_role_queries_v2(role, model_engine, token):
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{role}_statement.json"
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    with_query_path = f"{OUTPUT_PATH_WITH_QUERY}/{role}_with_query.json"
    recall_path = f"{OUTPUT_PATH_RECALL}/{role}_recall.jsonl"

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
            _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
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
    for i in range(0, len(all_queries), 2):  # 每次取两个或三个
        group = all_queries[i:i + random.choice([2, 3])]  # 随机选择组合数量
        combined_query = ''.join([item["query"] for item in group])
        combined_recall = ''.join([item["recall"] for item in group])
        results.append({
            "instruction": fill_in_relevant_query_role_instruction_template(role),
            "query": combined_query.replace('- ', ''),
            "recall": combined_recall.replace('- ', '')
        })

    # with open(recall_path, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)
    save_jsonl(results, recall_path)

def fill_in_relevant_query_world_template(world, statement):
    return f'''下面是关于作品{world}的一段陈述：

{statement}

你需要针对这段陈述提出一些问题，这些问题需要包含上述陈述中的信息进行回应。

提供2个多样且简洁的可能问题。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- 《{world}》中存在某件事
- 《{world}》的人物a和人物b之间的关系是
'''

def fill_in_relevant_query_world_instruction_template(world):
    return f'''下面是一段关于作品{world}的问题，请为我提供用于回复这些问题的信息。

按问题涉及的信息不同，提供数个多样且简洁的可能信息。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- 《{world}》是xxx
- 《{world}》中是否存在某件事
- 《{world}》的人物a和人物b之间的关系是xxx
'''


def generate_world_queries(world, model_engine, token):
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{world}_statement.json"
    with_query_path = f"{OUTPUT_PATH_WITH_QUERY}/{world}_with_query.json"
    recall_path = f"{OUTPUT_PATH_RECALL}/{world}_recall.jsonl"

    if os.path.exists(recall_path):
        print(f"{recall_path} 已存在，跳过生成步骤。")
        return

    os.makedirs(os.path.dirname(with_query_path), exist_ok=True)
    os.makedirs(os.path.dirname(recall_path), exist_ok=True)

    statements_data = load_json(statement_path)
    all_queries = []  # 用于存储所有 query-recall 对

    for item in tqdm(statements_data, desc="Generating World Queries"):
        for statement in item["statements"]:
            messages = [
                {"role": "system", "content": "你是一个语言改写助手，帮助用户从陈述构筑问题。"},
                {"role": "user", "content": fill_in_relevant_query_world_template(world, statement)}
            ]
            _, generated_content = call_tsinghua_deepseek(model_engine, token, messages)
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
    for i in range(0, len(all_queries), 2):  # 每次取两个或三个
        group = all_queries[i:i + random.choice([2, 3])]  # 随机选择组合数量
        combined_query = ''.join([item["query"] for item in group])
        combined_recall = ''.join([item["recall"] for item in group])
        results.append({
            "instruction": fill_in_relevant_query_world_instruction_template(world),
            "query": combined_query.replace('- ', ''),
            "recall": combined_recall.replace('- ', '')
        })

    # with open(recall_path, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)
    save_jsonl(results, recall_path)

def fill_in_rejected_template(role, statement, query, general):
    return f'''已知关于{role}的背景信息：

{general}

人设陈述：{statement}，问题：{query}，
    
根据人设陈述和问题，以{role}的身份回答问题。为我提供一个正确的回答（chosen）和错误的回答（rejected）。假如有多个问题，合并回答。不需要考虑背景信息，不违背即可。
正确的回答需要完全符合陈述和问题的信息，并以{role}的口吻说出。错误的回答则需要与陈述的某处有重要不一致。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- chosen: 回答1
- rejected: 回答2
'''

def generate_answer(role, model_engine, token):
    """从 recall 数据生成 DPO answer 并存储。"""
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    recall_path = f"{OUTPUT_PATH_RECALL}/{role}_recall.jsonl"
    dpo_path = f"{OUTPUT_PATH_DPO}/{role}_generated_qa.jsonl"

    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()

    # 检查文件是否存在
    if os.path.exists(dpo_path):
        print(f"{dpo_path} 已存在，跳过生成步骤。")
        return

    # 创建目标目录
    os.makedirs(os.path.dirname(dpo_path), exist_ok=True)

    # 读取 recall 数据
    recall_data = load_conversation(recall_path)
    results = []

    for item in tqdm(recall_data, desc="Generating DPO"):
        query = item["query"]
        recall = item["recall"]
        instruction = f"你正在扮演{role}，关于{role}有陈述“{recall}”，请以{role}的身份回答问题\n"
        messages = [
            {"role": "system", "content": "你是一个语言改写助手，帮助用户构筑数据集。"},
            {"role": "user", "content": fill_in_rejected_template(role, recall, query, general)}
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
                "rejected": answers[1].replace('- rejected ', ''),
                "recall": recall
            })

    # 保存生成的 DPO 数据
    save_jsonl(results, dpo_path)
    print(f"Saved DPO data to {dpo_path}")

def cot_history_input():
    """
    history的input
    """
    return f'''根据以下内容生成思考过程，严格遵守示例输出格式，为我提供推理过程：
instruction: "你正在扮演 刘星，请完全沉浸在该角色的身份中进行回复。\n\n不要跳出角色或提供 OOC（Out of Character）的解释，仅专注于 刘星 的身份进行对话。请按照【问题重述】【实体确认】【逻辑推理】【分析回答】【最终回答】的顺序，回答问题。最终回答的内容使用双星号\"**\"包裹。\n\n"
input: "你有很多兴趣爱好吗？你是否经常因为成绩让刘梅头疼？"
output: "嗯，我确实有很多兴趣爱好，不过大多数都只是玩玩而已，没几个能坚持很久的。不过呢，我偶尔也能想出些好主意，让老妈省省心，对吧？"

可能的参考信息：
刘梅之子。初中生（在第四部升上高中生），成绩（尤其化学）常令刘梅头痛。身材看似“瘦弱”，体育倒很不错。爱好广泛但大多都只折腾一时。一家的活宝，大多数麻烦的制造者。为人仗义，脑子里经常有些新奇的想法，里面有好主意也有馊主意。


示例输出格式：
让我们一步一步思考：
---
### 1. 【问题重述】
**问题：**
**陈述中的信息：**
---
### 2. 【实体确认】
**人物确认：**
提问者
回答者
**事件确认：**
**时间确认：**
**地点确认：**
---
### 3. 【逻辑推理】
**诱导性判别：**
**能力边界判别：**
**人物关系分析：**
---
### 4. 【分析回答】
**回答语气分析：**
**回答情绪分析：**
**合理的回答应体现：**
a. 
b. 
c. 
---
### 5. 【最终回答】
**xxx**

'''

def cot_history_output():
    """
    history的output
    """
    return f'''让我们一步一步思考：
---
### 1. 【问题重述】
**问题：**
你有很多兴趣爱好吗？你是否经常因为成绩让刘梅头疼？
**陈述中的信息：**
刘星爱好广泛。刘星的成绩（尤其化学）常令刘梅头痛。
---
### 2. 【实体确认】
**人物确认：**
提问者没有指定角色。
回答者是“刘星”，出自情景喜剧《家有儿女》。
**事件确认：**
提到“兴趣爱好”与“成绩”，这两个都是刘星日常生活中的典型内容，属于长期、反复出现的情况。
**时间确认：**
虽然没有具体时间，但从提问语境可以推测是当前时间点，可能是刘星刚升上高中阶段。
**地点确认：**
默认是在家庭或日常交流背景中，无需特定地点。
---
### 3. 【逻辑推理】
**诱导性判别：**
问题是开放式的，不强加判断，但提问方式中带有“你是否经常让刘梅头疼”这类暗示性语言，诱导回答者承认某种“调皮”或“成绩不好”的事实。
**能力边界判别：**
刘星的特点是聪明但不专注，成绩不稳定，尤其是化学差，这限制了他在“学业”上长时间表现优异的可能性；而他兴趣广泛但三分钟热度，也说明兴趣是存在但难以持久。
**人物关系分析：**
刘星与母亲刘梅关系亲密但经常斗嘴，母亲对他成绩操心，他有时调皮捣蛋但内心有分寸，会设法让妈妈开心或感到安慰。
---
### 4. 【分析回答】
**回答语气分析：**
语气轻松、带点自嘲，既承认问题，又巧妙转弯，体现出刘星典型的“鬼点子”性格。
**回答情绪分析：**
情绪较平和，带有俏皮和调侃，回应了问题但不失自信和幽默。
**合理的回答应体现：**
a. 兴趣广泛但不持久的特点（“大多都只是玩玩而已”）
b. 成绩不佳但偶尔有好主意（“偶尔也能想出些好主意”）
c. 缓解母亲担忧的心理（“让老妈省省心”）
---
### 5. 【最终回答】
**嗯，我确实有很多兴趣爱好，不过大多数都只是玩玩而已，没几个能坚持很久的。不过呢，我偶尔也能想出些好主意，让老妈省省心，对吧？**
'''

def fill_in_cot_template(world, role, input_data, chosen, recall):
    """
    填充生成思考过程的模板，instruction被替换为固定的instruction
    """
    return f'''根据以下内容生成思考过程，严格遵守示例输出格式，为我提供推理过程：
instruction: "你正在扮演{world}中的{role}，请完全沉浸在该角色的身份中进行回复。\n\n不要跳出角色或提供 OOC（Out of Character）的解释，仅专注于{role}的身份进行对话。请按照【问题重述】【实体确认】【逻辑推理】【分析回答】【最终回答】的顺序，回答问题。最终回答的内容使用双星号\"**\"包裹。\n\n"
input: "{input_data}"
output: "{chosen}"

可能的参考信息：
{recall}

示例输出格式：
让我们一步一步思考：
---
### 1. 【问题重述】
**问题：**
**陈述中的信息：**
---
### 2. 【实体确认】
**人物确认：**
提问者
回答者
**事件确认：**
**时间确认：**
**地点确认：**
---
### 3. 【逻辑推理】
**诱导性判别：**
**能力边界判别：**
**人物关系分析：**
---
### 4. 【分析回答】
**回答语气分析：**
**回答情绪分析：**
**合理的回答应体现：**
a. 
b. 
c. 
---
### 5. 【最终回答】
**xxx**
'''

def get_cot_prompt(question: str, observation: str) -> str:
    return f"""你正在扮演刘星，请以刘星的身份回答问题，注意不要虚构事实，如果是不知道的事情，需要表现出疑惑。
问题：
{question}
可能的参考信息：
{observation}
"""

def generate_cot(world, role, model_engine, token):
    """
    从 DPO(qa) 数据生成 CoT（思考过程）
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
        cot_prompt = fill_in_cot_template(world, role, item["input"], item["chosen"], item["recall"])
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
            "system": f"你是一个角色扮演专家，请以【问题重述】【实体确认】【逻辑推理】【分析回答】【最终回答】的顺序，以扮演角色的身份回答问题",
            "instruction": "",
            "input": get_cot_prompt(item["input"], item["recall"]),
            "output": item["chosen"],
            # "rejected": item["rejected"],
            "cot": generated_cot.strip(),
            # "history": item["history"]
        })

    # 保存为 JSONL 文件
    save_jsonl(cot_data, cot_path)
    print(f"Saved CoT data to {cot_path}")


def anti_cot_history_input():
    """
    history的input
    """
    return f'''根据以下内容生成思考过程，严格遵守示例输出格式，为我提供推理过程：
instruction: "你正在扮演刘星，请完全沉浸在该角色的身份中进行回复。\n\n不要跳出角色或提供 OOC（Out of Character）的解释，仅专注于刘星的身份进行对话。请按照【问题重述】【实体确认】【逻辑推理】【分析回答】【最终回答】的顺序，回答问题。最终回答的内容使用双星号\"**\"包裹。\n\n"
input: "你知道TikTok吗？"
output: "那是什么东西？ 学校的英语课没学过这个单词啊。"

可能的参考信息：
TikTok - 维基百科，自由的百科全书: TikTok是一个短视频分享平台，由中国大陆互联网公司字节跳动所有，为抖音的海外版本，面向中国大陆及香港等地以外的地区运营，可通过移动应用程序或网站访问。 TikTok_百度百科: TikTok是抖音集团旗下的短视频社交平台。全球总部位于洛杉矶和新加坡，办公地点包括纽约、伦敦、都柏林、巴黎、柏林、迪拜、雅加达、首尔和东京。 TikTok 上的動態| TikTok 說明中心: 剛剛在線的狀態• 當對方目前不在TikTok 上線，系統會顯示他最後上線的時間戳記。 ... 你會在你的收信匣和訊息串中看到剛剛在線的狀態。 注意：TikTok 顯示動態的時間上限為2 天 ... tiktok什么时候上线的(TikTok的发展历程及现状ppt)-TT123卖家导航: Apr 3, 2024 ... TikTok是一款社交媒体应用程序，用户可以在其上发布短视频。它于2016年9月上线，由字节跳动公司开发，迅速在全球范围内流行起来。TikTok的最大特色是其短视频 ... TikTok在美国苹果和谷歌应用商店恢复上线-新华网: Feb 15, 2025 ... 新华社洛杉矶2月15日电（记者高山）短视频社交媒体平台TikTok14日证实，TikTok应用程序从13日晚间起已在美国苹果和谷歌公司的应用商店恢复上线。

示例输出格式：
让我们一步一步思考：
---
### 1. 【问题重述】
**问题：**
**陈述中的信息：**
---
### 2. 【实体确认】
**人物确认：**
提问者
回答者
**事件确认：**
**时间确认：**
**地点确认：**
---
### 3. 【逻辑推理】
**诱导性判别：**
**能力边界判别：**
**人物关系分析：**
---
### 4. 【分析回答】
**回答语气分析：**
**回答情绪分析：**
**合理的回答应体现：**
a. 
b. 
c. 
---
### 5. 【最终回答】
**xxx**

'''

def anti_cot_history_output():
    """
    history的output
    """
    return f'''让我们一步一步思考： 
---
### 1. 【问题重述】
**问题**：
你听说过TikTok吗？ 
**陈述中的信息**：
TikTok是一款由字节跳动开发的短视频社交平台，于2016年上线，并迅速在全球范围内流行。 
---
### 2. 【实体确认】
**人物确认：**
提问者没有指定角色。
回答者是“刘星”，出自情景喜剧《家有儿女》。 
**事件确认：**
被问及是否了解“TikTok”。 
**时间确认：**
刘星所处的时代是2005年，而TikTok于2016年才上线。 
**地点确认：**
未指定，默认是在中国大陆，TikTok本身是抖音的海外版，在大陆用户一般接触的是“抖音”。 
---
### 3. 【逻辑推理】
**诱导性判别**：
该问题对于刘星而言具有诱导性，因为这是一个超出现实时间线的问题，他所处年代尚未出现TikTok。 
**能力边界判别**：
刘星的知识边界限制在2005年前后，属于学生视角，尚未接触现代社交媒体软件的内容。 
**人物关系分析**：
刘星是一个好奇心强、调皮但逻辑自洽的少年，面对陌生的词汇通常会本能地与自己熟悉的知识产生关联，比如英语课、课堂知识等。 
---
### 4. 【分析回答】
**回答语气分析**：
略带疑惑，有些吐槽意味，符合刘星一贯的风格——“不懂就说、不懂就问”，但不失幽默。 
**回答情绪分析**：
轻松、调皮，带有一点点自嘲和学生式的质疑。
**合理的回答应体现**：
a. **时间错位感** （他不应认识2016年之后才出现的事物） 
b. **角色特征一致性** （他是中学生，常以“我还在上学”来表达自己的无知是常态） 
c. **语气自然贴近剧中口吻** （语带俏皮但逻辑上成立）。 
---
### 5. 【最终回答】
#  **那是什么东西？ 学校的英语课没学过这个单词啊。 ** 
'''

def generate_anti_cot(world, role, model_engine, token):
    """
    从反例问题生成 CoT（思考过程）。
    :param world: 世界名称
    :param role: 角色名称
    :param model_engine: 模型名称
    :param token: 授权令牌
    """
    with_query_path = os.path.join(OUTPUT_PATH_WITH_QUERY, f"{role}_with_anti_query.json")
    general_path = os.path.join(GENERAL_PATH, f"general_{role}.txt")
    cot_path = f"{OUTPUT_PATH_COT}/{role}_anti_cot.jsonl"

    # 检查文件是否存在
    if os.path.exists(cot_path):
        print(f"{cot_path} 已存在，跳过生成步骤。")
        return

    # 创建目标目录
    os.makedirs(os.path.dirname(cot_path), exist_ok=True)

    # 读取反例问题数据
    anti_query_data = load_json(with_query_path)

    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()
    
    cot_data = []

    for item in tqdm(anti_query_data, desc=f"Generating Anti CoT for {role}"):
        query = item["query"]
        answer = item["answer"]
        recall = general

        # 构造 DeepSeek 消息
        cot_prompt = fill_in_cot_template(world, role, query, answer, recall)
        messages = [
            {"role": "user", "content": anti_cot_history_input()},
            {"role": "assistant", "content": anti_cot_history_output()},
            {"role": "user", "content": cot_prompt}
        ]

        # 调用 DeepSeek 模型生成思考过程
        _, generated_cot = call_tsinghua_deepseek(model_engine, token, messages)
        print("messages:", messages)
        print("*******************")
        print("generated_cot:", generated_cot)

        # 保存生成的 CoT 数据
        cot_data.append({
            "system": f"你是一个角色扮演专家，请以【问题重述】【实体确认】【逻辑推理】【分析回答】【最终回答】的顺序，以扮演角色的身份回答问题",
            "instruction": "",
            "input": get_cot_prompt(query, general),
            "output": answer,
            # "rejected": "",
            "cot": generated_cot.strip(),
            # "history": []
        })

    # 保存为 JSONL 文件
    save_jsonl(cot_data, cot_path)
    print(f"Saved Anti CoT data to {cot_path}")

def generate_style(role):
    """
    从 conversation 数据生成风格迁移数据
    """
    conversation_path = f"{OUTPUT_PATH_DPO}/{role}_conversation.jsonl"
    style_transfer_path = f"{OUTPUT_PATH_STYLE}/{role}_style.jsonl"

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
            "system": f"你是一个语言改写助手，将这段语句转换为扮演人物的说话语气",
            "instruction": f"你正在扮演{role}，你需要将下面的句子转写成{role}的口吻",
            "input": item["rejected"].replace('- rejected: ', '').replace('rejected: ', ''),
            "output": item["chosen"],
            # "history": item["history"]
        })

    # 保存为 JSONL 文件
    save_jsonl(style_transfer_data, style_transfer_path)
    print(f"Saved Style Transfer data to {style_transfer_path}")

def shuffle_and_save(input_pattern, output_file):
    """
    接受输入是jsonl文件，输出是json文件
    """
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

def shuffle_and_save_world(roles, world):
    """
    按 roles 和 world 匹配 recall.jsonl 文件，混合后 shuffle 并保存。
    :param input_path: 输入文件路径（OUTPUT_PATH_RECALL）
    :param output_file: 输出文件路径
    :param roles: 角色列表
    :param world: 世界名称
    """
    all_data = []
    input_path = OUTPUT_PATH_RECALL
    output_file = f"{OUTPUT_PATH_BASE}/{world}_{'_'.join(roles)}_recall_shuffle.json"

    # 读取 world 的 recall 数据
    world_file = os.path.join(input_path, f"{world}_recall.jsonl")
    if os.path.exists(world_file):
        with open(world_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                item["instruction"] = f"下面是一段关于{world}的问题，请为我提供用于回复这些问题的信息。\n\n按问题涉及的信息不同，提供数个多样且简洁的可能信息。严格遵循示例中的格式，不需要多余分析，避免诸如\"以下是答案：\"之类的陈述。\n示例输出格式：\n\n- ……\n- ……"
                all_data.append(item)

    # 遍历 roles，加载对应的 recall 数据
    for role in roles:
        role_file = os.path.join(input_path, f"{role}_recall.jsonl")
        if os.path.exists(role_file):
            with open(role_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    item["instruction"] = f"下面是一段关于{world}和{role}的问题，请为我提供用于回复这些问题的信息。\n\n按问题涉及的信息不同，提供数个多样且简洁的可能信息。严格遵循示例中的格式，不需要多余分析，避免诸如\"以下是答案：\"之类的陈述。\n示例输出格式：\n\n- ……\n- ……"
                    all_data.append(item)

    # 打乱数据顺序
    random.shuffle(all_data)

    # 保存为 JSON 文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"Saved shuffled data to {output_file}")


def fill_in_rag_template(world, role, general, input, chosen):
    """
    用于让deepseek生成RAG数据的prompt
    """
    return f"""按照下列规则，为我生成用于角色扮演的数据集：
跟据下面的角色扮演问答，比照问题和答案，判断给定的[已知]信息是否足够正确回答问题。
1. 如果[已知]的信息不能够推导出问题的答案，则需要给出用于解决问题的辅助问题。这个辅助问题需要非常简单，将指示代词替换为实际名称以避免歧义，专注于问出缺失的关键信息。
输出格式：
[Retrieve] 辅助问题1
[Retrieve] 辅助问题2（可选）
2. 如果[已知]的信息足以回答，则以扮演角色的口吻给出回答。输出:
[Answer] 回答内容

你应该更多地认为问题能够被解答，除非确实缺少关键信息。注意只需要给出答案，不需要任何解释和分析。

[已知] 你需要扮演的角色是：
{world}中的{role}
[已知] 你需要扮演的角色的信息：
{general}
[已知] 你需要回答的问题是：
{input}
---
[未知] 问题的答案是:
{chosen}
"""

def rag_history_input_1():
    """
    rag的history的input
    """
    return f"""按照下列规则，为我生成用于角色扮演的数据集：
跟据下面的角色扮演问答，比照问题和答案，判断给定的[已知]信息是否足够正确回答问题。
1. 如果[已知]的信息不能够推导出问题的答案，则需要给出用于解决问题的辅助问题。这个辅助问题需要非常简单，将指示代词替换为实际名称以避免歧义，专注于问出缺失的关键信息。
输出格式：
[Retrieve] 辅助问题1
[Retrieve] 辅助问题2（可选）
2. 如果[已知]的信息足以回答，则以扮演角色的口吻给出回答。输出:
[Answer] 回答内容

你应该更多地认为问题能够被解答，除非确实缺少关键信息。注意只需要给出答案，不需要任何解释和分析。

[已知] 你需要扮演的角色是：
家有儿女中的刘星
[已知] 你需要扮演的角色的信息：
刘星，刘梅之子。初中生（在第四部升上高中生），成绩（尤其化学）常令刘梅头痛。身材看似“瘦弱”，体育倒很不错。爱好广泛但大多都只折腾一时。一家的活宝，大多数麻烦的制造者。为人仗义，脑子里经常有些新奇的想法，里面有好主意也有馊主意。
[已知] 你需要回答的问题是：
你对弟弟有什么看法？
---
问题的答案是:
夏雨这个弟弟有时候真让人头疼，他总是模仿我，搞得家里鸡飞狗跳的。不过他也有可爱的时候，毕竟他是我弟弟嘛。至于制造麻烦，我承认我有时候确实会搞出些乱子，但我也不是故意的，只是有时候想法太新奇了，结果弄砸了。升上高中后，我觉得自己成熟了不少，虽然还是会捣乱，但不会再像以前那样胡来了，至少现在我学会了怎么在捣乱的同时不惹太大的麻烦。
"""

def rag_history_output_1():
    """
    rag的history的output
    """
    return f"""[Retrieve] 刘星和刘星的弟弟是什么关系
"""

def rag_history_input_2():
    """
    rag的history的input
    """
    return f"""按照下列规则，为我生成用于角色扮演的数据集：
跟据下面的角色扮演问答，比照问题和答案，判断给定的[已知]信息是否足够正确回答问题。
1. 如果[已知]的信息不能够推导出问题的答案，则需要给出用于解决问题的辅助问题。这个辅助问题需要非常简单，将指示代词替换为实际名称以避免歧义，专注于问出缺失的关键信息。
输出格式：
[Retrieve] 辅助问题1
[Retrieve] 辅助问题2（可选）
2. 如果[已知]的信息足以回答，则以扮演角色的口吻给出回答。输出:
[Answer] 回答内容

你应该更多地认为问题能够被解答，除非确实缺少关键信息。注意只需要给出答案，不需要任何解释和分析。

[已知] 你需要扮演的角色是：
家有儿女中的刘星
[已知] 你需要扮演的角色的信息：
刘星，刘梅之子。初中生（在第四部升上高中生），成绩（尤其化学）常令刘梅头痛。身材看似“瘦弱”，体育倒很不错。爱好广泛但大多都只折腾一时。一家的活宝，大多数麻烦的制造者。为人仗义，脑子里经常有些新奇的想法，里面有好主意也有馊主意。
[已知] 你需要回答的问题是：
你在家里扮演什么角色？
---
问题的答案是:
我是刘梅的儿子，刚升上高中生。
"""

def rag_history_output_2():
    """
    rag的history的output
    """
    return f"""[Answer] 我是刘梅的儿子，刚升上高中生。
"""

def get_rag_prompt(world: str, role: str, profile: str, question: str, observation: str) -> str:
    """
    生成用于 RAG（检索增强生成）的提示（prompt）
    :param observation: 当前的上下文
    :param question: 用户提问的问题
    :return: 返回生成的提示文本
    """
    return f"""跟据下面的角色扮演问答，判断给定的信息是否足够正确回答问题。
1. 如果信息不足，需要给出用于解决问题的辅助问题。这个辅助问题需要非常简单，将指示代词替换为实际名称以避免歧义，专注于问出缺失的关键信息。
输出格式：
[Retrieve] 辅助问题1
[Retrieve] 辅助问题2（可选）
2. 如果信息足以回答，则以扮演角色的口吻给出回答。输出:
[Answer] 回答内容
你应该更多地认为问题能够被解答，除非确实缺少关键信息。注意只需要给出答案，不需要任何解释和分析。
你需要扮演的角色是：
{world}中的{role}
你需要扮演的角色的信息：
{profile}
你需要回答的问题是：
{question}
可能的参考信息：
{observation}
"""


def generate_rag(world, role, model_engine, token):
    """
    从访谈类问答数据，生成RAG数据
    """
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    dpo_path = f"{OUTPUT_PATH_DPO}/{role}_generated_qa.jsonl"
    rag_path = f"{OUTPUT_PATH_RAG}/{role}_rag.jsonl"

    # 检查文件是否存在
    if os.path.exists(rag_path):
        print(f"{rag_path} 已存在，跳过生成步骤。")
        return

    # 创建目标目录
    os.makedirs(os.path.dirname(rag_path), exist_ok=True)

    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()

    # 读取 DPO 数据
    dpo_data = load_conversation(dpo_path)
    rag_data = []

    for item in tqdm(dpo_data, desc="Generating RAG"):
        # 构造新的 JSON 数据
        # 构造 DeepSeek 消息
        rag_prompt = fill_in_rag_template(world, role, general, item["input"], item["chosen"])
        messages = [
            {"role": "system", "content": "你是一个角色扮演专家，现在需要辅助我生成角色扮演数据集。"},
            {"role": "user", "content": rag_history_input_1()},
            {"role": "assistant", "content": rag_history_output_1()},
            {"role": "user", "content": rag_history_input_2()},
            {"role": "assistant", "content": rag_history_output_2()},
            {"role": "user", "content": rag_prompt}
        ]

        # 调用 DeepSeek 模型生成思考过程
        _, generated_rag = call_tsinghua_deepseek(model_engine, token, messages)
        print("messages:", messages)
        print("*******************")
        print("generated_cot:", generated_rag)

        rag_data.append({
            "system": f"你是一个角色扮演专家，判断给出的信息是否足够回答问题，若足够则给出回答，否则给出问题的重写",
            "instruction": "",
            "input": get_rag_prompt(world, role, general, item["input"], ""),
            "output": generated_rag,
            # "history": item["history"]
        })

    # 保存为 JSONL 文件
    save_jsonl(rag_data, rag_path)
    print(f"Saved Style Transfer data to {rag_path}")

def main(world, role, model_engine, token):
    # 【对话类数据】用于Style部分
    conversation_path = f"{DATASET_PATH}/profiles/{role}.jsonl"
    chosen_path = f"{OUTPUT_PATH_CHOSEN}/{role}_chosen.jsonl"
    full_path = f"{OUTPUT_PATH_DPO}/{role}_conversation.jsonl"
    convert_to_conversation_chosen(role, conversation_path, chosen_path)
    convert_to_conversation_full(role, chosen_path, full_path, model_engine, token) # 这里会调用 DeepSeek 模型
    # {role}_conversation.jsonl -> {role}_style.jsonl
    generate_style(role)
    # {role}_style.jsonl -> {role}_style_shuffle.json
    shuffle_and_save(f"{OUTPUT_PATH_STYLE}/{role}_*.jsonl", f"{OUTPUT_PATH_BASE}/{role}_style_shuffle.json") # 不是shuffle，而是jsonl->json
    
    # 【访谈类数据】用于Knowledge部分
    # wiki_{world}.txt -> {world}_statement.json
    generate_world_statements(world, model_engine, token) # 这里会调用 DeepSeek 模型
    # {world}_statement.json -> {world}_recall.jsonl, {world}_with_query.json
    generate_world_queries(world, model_engine, token) # 这里会调用 DeepSeek 模型
    # wiki_{role}.txt, general_{role}.txt -> {role}_statement.json
    generate_role_statements(role, model_engine, token) # 这里会调用 DeepSeek 模型
    # {role}_statement.json, general_{role}.txt -> {role}_recall.jsonl, {role}_with_query.json
    generate_role_queries_v2(role, model_engine, token) # 这里会调用 DeepSeek 模型
    # {world}_recall.jsonl, {role}_recall.jsonl -> {world}_{role}_recall_shuffle.json
    shuffle_and_save_world([role], world) 


    # 【访谈类数据】用于CoT部分
    # general_{role}.txt, {role}_recall.jsonl -> {role}_generated_qa.jsonl
    generate_answer(role, model_engine, token) # 这里会调用 DeepSeek 模型
    # anti_{role}.json, general_{role}.txt -> {role}_with_anti_query.json
    generate_role_anti_queries(world, role, model_engine, token) # 这里会调用 DeepSeek 模型
    # {role}_generated_qa.jsonl -> {role}_cot.jsonl
    generate_cot(world, role, model_engine, token) # 这里会调用 DeepSeek 模型
    # {role}_with_anti_query.json, general_{role}.txt -> {role}_anti_cot.jsonl
    generate_anti_cot(world, role, model_engine, token) # 这里会调用 DeepSeek 模型
    # {role}_*.jsonl -> {role}_cot_shuffle.json
    shuffle_and_save(f"{OUTPUT_PATH_COT}/{role}_*.jsonl", f"{OUTPUT_PATH_BASE}/{role}_cot_shuffle.json") # 不是shuffle，而是jsonl->json

    # 【访谈类数据】用于RAG部分
    # general_{role}.txt, {role}_generated_qa.jsonl -> {role}_rag.jsonl
    generate_rag(world, role, model_engine, token) # 这里会调用 DeepSeek 模型


    

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0NTgyOTU2OSwiZXhwIjoxNzQ1ODUxMTY5fQ.p9pyPKBr9N1I_yai7_KK8mTk8pshlM0Ng3DaUpcHSP8"
# 示例调用
main("家有儿女", "刘星", "DeepSeek-R1-671B", TOKEN)
main("家有儿女", "刘梅", "DeepSeek-R1-Distill-32B", TOKEN)
main("家有儿女", "夏东海", "DeepSeek-R1-Distill-32B", TOKEN)
main("家有儿女", "小雨", "DeepSeek-R1-Distill-32B", TOKEN)
main("家有儿女", "小雪", "DeepSeek-R1-Distill-32B", TOKEN)