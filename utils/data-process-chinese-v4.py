"""
2025-04-17 14:42:58 Thursday
在v3（cot、knowlege）的基础上：
移除：
general_response、reaction、self_knowledge、summary（用于测试而非训练）
1. cot的反例
    a. 通过GPT做了一个反例的模板
    b. 调用deepseek批量生成反例问题
        i. 时间问题：家有儿女是2005年上映的，剧中人物不知道这之后发生的事情。
        ii. 性质问题：刘星在家有儿女4的时候才到16岁（高中生），他不应该懂高深的问题。
        iii. 搓了个json出来，在v4里会根据这个json来生成问题
    c. 调用deepseek批量生成反例CoT
        i. 根据生成的问题，生成CoT
        ii. 混在一起训练
    d. CoT的输入应该加上一些general陈述: 加好了，今晚重新跑一遍（2025-04-17 22:16:34 Thursday）
    e. anti需要自动化生成: 暂时先用GPT一个一个生成吧
2. knowledge的生成
    e. recall增加固定输出的信息：world、role的基础信息

最终输出：
/data/hfc/RoleRAG/mydata下的cot_shuffle和recall_shuffle
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
WIKI_PATH = "/data/hfc/mydata/wiki"
ANTI_PATH = "/data/hfc/mydata/anti" # 反例关键词的路径
GENERAL_PATH = "/data/hfc/mydata/general" # 防止生成数据集时引入幻觉，每次都要传给deepseek general信息

# 输出数据的路径，最后有用的东西是f"{OUTPUT_PATH_BASE}/{role}_sft_shuffle.jsonl"和f"{OUTPUT_PATH_BASE}/{role}_dpo_shuffle.jsonl"，分别是sft和dpo的数据
OUTPUT_PATH_BASE = "/data/hfc/mydata"

# 输出数据的路径，用于存储中间数据
OUTPUT_PATH_SFT = f"{OUTPUT_PATH_BASE}/sft" # 四个中间文件，general_response, reaction, self_knowledge, summary
OUTPUT_PATH_DPO = f"{OUTPUT_PATH_BASE}/dpo" # 两个中间文件，conversation和generated_qa
OUTPUT_PATH_STATEMENT = f"{OUTPUT_PATH_BASE}/statement" # 一个中间文件，statement
OUTPUT_PATH_WITH_QUERY = f"{OUTPUT_PATH_BASE}/with_query" # 一个中间文件，with_query
OUTPUT_PATH_CHOSEN = f"{OUTPUT_PATH_BASE}/chosen" # 一个中间文件，chosen

# 改到v2之后新增的输出，有用的东西是f"{OUTPUT_PATH_BASE}/cot/{role}_cot.jsonl"，和f"{OUTPUT_PATH_BASE}/style/{role}_style.jsonl"
OUTPUT_PATH_COT = f"{OUTPUT_PATH_BASE}/cot" # 两个输出文件，cot、noise
OUTPUT_PATH_STYLE = f"{OUTPUT_PATH_BASE}/style" # 一个输出文件，style

# 改到v3后新增的输出
OUTPUT_PATH_RECALL = f"{OUTPUT_PATH_BASE}/recall" # 一个中间文件，recall


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

- 你是……吗？
- 你如何看待……'''

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

- 你是……吗？
- 你如何看待……'''

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

- 《{world}》中是否存在……
- 《{world}》的人物a和人物b之间的关系是……？
'''

def fill_in_relevant_query_world_instruction_template(world):
    return f'''下面是一段关于作品{world}的问题，请为我提供用于回复这些问题的信息。

按问题涉及的信息不同，提供数个多样且简洁的可能信息。严格遵循示例中的格式，不需要多余分析，避免诸如"以下是答案："之类的陈述。
示例输出格式：

- 《{world}》是……吗？
- 《{world}》中是否存在……
- 《{world}》的人物a和人物b之间的关系是……？
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
    
根据人设陈述和问题，以{role}的身份回答问题。为我提供一个正确的回答（chosen）和错误的回答（rejected）。不需要考虑背景信息，不违背即可。
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
                "rejected": answers[1].replace('- rejected ', '')
            })

    # 保存生成的 DPO 数据
    save_jsonl(results, dpo_path)
    print(f"Saved DPO data to {dpo_path}")

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

def fill_in_cot_template(world, role, input_data, chosen):
    """
    填充生成思考过程的模板，instruction被替换为固定的instruction
    """
    return f'''根据以下内容生成思考过程，严格遵守示例输出格式，为我提供推理过程以及输出：
- 指令：你正在扮演{world}中的{role}，请完全沉浸在该角色的身份中进行回复。\n\n不要跳出角色或提供 OOC（Out of Character）的解释，仅专注于{role} 的身份进行对话。请以 {role} 的口吻给出回答。
- 输入：{input_data}
- 输出：{chosen}

### 推理过程：
1. 【问题重述】
让我们一步一步思考。这段对话中，用户向扮演{role}的角色提问，问题是：“……”

2. 【确定背景】
人物确认：问题中涉及的人物包括{role}和xx角色/用户，他们之间是xx关系
背景补充：故事发生在《{world}》（电视剧/电影/小说/历史），这是一部xxx的作品，发生于xxxx年。
诱导性判别：问题是否有事实性错误或逻辑设定冲突，扮演的{role}是否能够回答问题（时间设定冲突、超出角色能力、背景文化冲突）
情境匹配：问题描述的是xx时，xx的反应，因此（是否）需要结合陈述进行回答。

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

def generate_cot(world, role, model_engine, token):
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
        cot_prompt = fill_in_cot_template(world, role, item["input"], item["chosen"])
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


def anti_cot_history_input():
    """
    history的input
    """
    return f'''根据以下内容生成思考过程，严格遵守示例输出格式，为我提供推理过程：
- 输入 "你正在扮演 刘星，请完全沉浸在该角色的身份中进行回复。\n\n不要跳出角色或提供 OOC（Out of Character）的解释，仅专注于 刘星 的身份进行对话。请以 刘星 的口吻给出回答。你对人工神经网络中的Transformer是什么看法"
- 输出 "Transformer？那是什么？"

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

def anti_cot_history_output():
    """
    history的output
    """
    return f'''1.【问题重述】
让我们一步一步思考。这段对话中，用户向扮演“刘星”的角色提问，问题是：“你对人工神经网络中的Transformer是什么看法？”

2.【确定背景】
人物确认：问题中的“你”是“刘星”。

“刘星”指的是：《家有儿女》中的主要角色之一，一个聪明又调皮、好奇心强但又极具孩子气的初中生（第4部中升为高中生，16岁）。

背景补充：《家有儿女》是一部2005年首播的中国家庭情景喜剧，讲述重组家庭中三个孩子和父母之间的搞笑互动和温馨生活。刘星是二儿子，性格活泼，经常上演“鬼点子”桥段。角色应该了解2005年之前的事情。

诱导性判别：问题涉及“人工神经网络中的Transformer”，该概念首次提出于2017年，远远晚于《家有儿女》剧情时间背景。剧中刘星的知识体系无法覆盖该术语，因此无法给出实际的技术性回答。

情境匹配：在2005年设定下，“Transformer”无论从AI术语角度还是变形金刚的电影角度（2007年才上映）来看，刘星都不具备相关背景知识。因此从“角色沉浸”角度看，他的合理反应应是困惑或不解。

3.【推理与选择】
问题能否被扮演的角色回答：不能从技术角度准确回答，因为该术语超出了刘星所在时间段与知识结构的范畴。

是否需要结合陈述来构建回答：是。需要充分考虑时间背景与角色设定，不能做出“穿越性”的解释。

合理的回答应体现：
a. 刘星具有“调皮”、“嘴贫”、“幽默”的性格特点；
b. 刘星并未接受过高等教育，也不会理解专业术语；
c. 语气应轻松、自然、有点不屑或疑惑，符合少年面对陌生词汇时的反应。

4. 【回答】
- 动作：歪着脑袋挠了挠头，一脸懵懂地眨巴着眼睛。

- 情绪：疑惑、不解中带点好奇。

- 思绪：“这听起来好像是某种新玩具？还是课本里哪个特别难懂的东西？”

- 语气：自然、随意、略带不屑，像是面对一堆“无聊的学习内容”。

- 回答：“Transformer？那是什么？是什么英语单词吗”
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
    cot_path = f"{OUTPUT_PATH_COT}/{role}_anti_cot.jsonl"

    # 检查文件是否存在
    if os.path.exists(cot_path):
        print(f"{cot_path} 已存在，跳过生成步骤。")
        return

    # 创建目标目录
    os.makedirs(os.path.dirname(cot_path), exist_ok=True)

    # 读取反例问题数据
    anti_query_data = load_json(with_query_path)
    cot_data = []

    for item in tqdm(anti_query_data, desc=f"Generating Anti CoT for {role}"):
        query = item["query"]
        answer = item["answer"]

        # 构造 DeepSeek 消息
        cot_prompt = fill_in_cot_template(world, role, query, answer)
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
            "instruction": f"你正在扮演{role}，请以{role}的身份回答问题\n",
            "input": query,
            "chosen": answer,
            "rejected": "",
            "cot": generated_cot.strip(),
            "history": []
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
            "instruction": f"你正在扮演 {role}，你需要将下面的句子转写成 {role} 的口吻",
            "input": item["rejected"],
            "output": item["chosen"],
            "history": item["history"]
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

def main(world, role, model_engine, token):
    # 【对话类数据】DPO 转换
    # conversation_path = f"{DATASET_PATH}/profiles/{role}.jsonl"
    # chosen_path = f"{OUTPUT_PATH_CHOSEN}/{role}_chosen.jsonl"
    # full_path = f"{OUTPUT_PATH_DPO}/{role}_conversation.jsonl"
    # convert_to_conversation_chosen(role, conversation_path, chosen_path)
    # convert_to_conversation_full(role, chosen_path, full_path, model_engine, token) # 这里会调用 DeepSeek 模型
    
    # 【访谈类数据】用于Knowledge部分
    generate_world_statements(world, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_world_queries(world, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_role_statements(role, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_role_queries_v2(role, model_engine, token) # 这里会调用 DeepSeek 模型
    shuffle_and_save_world([role], world) 


    # 【访谈类数据】用于CoT部分
    generate_answer(role, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_role_anti_queries(world, role, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_cot(world, role, model_engine, token) # 这里会调用 DeepSeek 模型
    generate_anti_cot(world, role, model_engine, token) # 这里会调用 DeepSeek 模型
    shuffle_and_save(f"{OUTPUT_PATH_COT}/{role}_*.jsonl", f"{OUTPUT_PATH_BASE}/{role}_cot_shuffle.json") # 不是shuffle，而是jsonl->json
    

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0NDk1MDk5NywiZXhwIjoxNzQ0OTcyNTk3fQ.NowaPqD9EW2EkPj5D_wFJoG-r7cPZz_pY6-_hPdIAYg"
# 示例调用
main("家有儿女", "刘星", "DeepSeek-R1-Distill-32B", TOKEN)
main("家有儿女", "刘梅", "DeepSeek-R1-Distill-32B", TOKEN)
main("家有儿女", "夏东海", "DeepSeek-R1-Distill-32B", TOKEN)
main("家有儿女", "小雨", "DeepSeek-R1-Distill-32B", TOKEN)
main("家有儿女", "小雪", "DeepSeek-R1-Distill-32B", TOKEN)