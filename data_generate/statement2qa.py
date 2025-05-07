"""
** wiki -> statement -> qa_statement(75) 
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
import re



WIKI_PATH = "/data/hfc/RoleRAG/data0506/input/wiki"
GENERAL_PATH = "/data/hfc/RoleRAG/data0506/input/general"
PROCESS_PATH_BASE = "/data/hfc/RoleRAG/data0506/process"
OUTPUT_PATH_STATEMENT = f"{PROCESS_PATH_BASE}/statement"
OUTPUT_PATH_QA_STATEMENT = f"/data/hfc/RoleRAG/data0506/qa/qa_statement"

def fill_in_relevant_query_role_template(character, statement, general):
    return f'''已知关于{character}的背景信息：

{general}

人设陈述：{statement}

你需要对{character}提出一些问题，这些问题需要包含上述人设陈述中的信息进行回应。不需要涉及背景信息，保持不违背即可。

提供1至3个多样且简洁的问题-回答对，这些问题将谈话对象视为{character}，且不包含名字；回答者以{character}的身份回答。 严格遵循示例中的格式(json数组)，不需要多余分析，避免诸如"以下是答案："之类的陈述。

示例输出格式：
[{{
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
}}]
'''

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def statement2qa(world, role, apikey, model_engine="gpt-4o-mini"):
    """
    根据角色陈述生成问答对，并保存到指定路径
    :param world: 世界名称（如"家有儿女"）
    :param role: 角色名称
    :param model_engine: GPT模型名称
    """
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=apikey,
        base_url="https://api.chatanywhere.tech/v1"
        # base_url="https://api.chatanywhere.org/v1"
)
    statement_path = f"{OUTPUT_PATH_STATEMENT}/{role}_statement.json"
    general_path = f"{GENERAL_PATH}/general_{role}.txt"
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_PATH_QA_STATEMENT, exist_ok=True)
    
    # 读取通用背景信息
    with open(general_path, 'r', encoding='utf-8') as f:
        general = f.read().strip()
    
    # 读取角色陈述数据
    statements_data = load_json(statement_path)
    
    # 用于存储生成的问答对
    all_qa_pairs = []
    
    # 随机打乱陈述顺序以获得多样性
    all_statements = []
    for item in statements_data:
        for statement in item["statements"]:
            all_statements.append(statement)
    
    random.shuffle(all_statements)
    
    # 从陈述中生成问答对
    generated_count = 0
    pbar = tqdm(total=len(all_statements), desc=f"Generating QA for {role}")
    
    for statement_index, statement in enumerate(all_statements):
        # 定期保存中间结果
        # if statement_index > 0 and statement_index % 5 == 0 and all_qa_pairs:
        #     # 保存中间结果
        #     temp_count = len(all_qa_pairs)
        #     temp_output_path = f"{OUTPUT_PATH_QA_STATEMENT}/{world}_{role}_qa_temp_{temp_count}.jsonl"
        #     with open(temp_output_path, "w", encoding="utf-8") as f:
        #         json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
        #     print(f"\n[进度保存] 已处理 {statement_index}/{len(all_statements)} 个陈述，生成 {temp_count} 个问答对")
        
        # 构造请求
        messages = [
            {"role": "system", "content": "你是一个语言改写助手，帮助用户从陈述构筑问题和回答。"},
            {"role": "user", "content": fill_in_relevant_query_role_template(role, statement, general)}
        ]
        
        print(f"\n--- 处理陈述 [{statement_index+1}/{len(all_statements)}]: {statement[:50]}...")
        
        # 重试机制
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 使用GPT生成问答对，添加超时参数
                response = client.chat.completions.create(
                    model=model_engine,
                    messages=messages,
                    temperature=0.8,
                    timeout=60  # 60秒超时
                )
                generated_content = response.choices[0].message.content.strip()
                
                print(f"--- 生成内容: {generated_content[:100]}...")
                
                # 解析JSON格式的响应
                try:
                    # 尝试提取JSON部分
                    json_content = generated_content
                    # 如果内容被其他文本包围，尝试提取[]之间的内容
                    if not json_content.startswith('['):
                        json_match = re.search(r'\[\s*{.*}\s*\]', json_content, re.DOTALL)
                        if json_match:
                            json_content = json_match.group(0)
                    
                    qa_pairs = json.loads(json_content)
                    
                    # 确保获取到的是列表
                    if not isinstance(qa_pairs, list):
                        print(f"警告: 预期列表但得到 {type(qa_pairs)}")
                        retry_count += 1
                        continue
                        
                    # 处理每个问答对
                    qa_added = False
                    for qa_pair in qa_pairs:
                        if "question" in qa_pair and "answer" in qa_pair:
                            question = qa_pair["question"]
                            answer = qa_pair["answer"]
                            
                            # 格式化并添加到结果中
                            all_qa_pairs.append({
                                "question": question,
                                "answer": answer,
                                "retrieve": statement
                            })
                            
                            generated_count += 1
                            qa_added = True
                    
                    if qa_added:
                        # 成功处理，跳出重试循环
                        break
                    else:
                        print("警告: 未找到有效的问答对")
                        retry_count += 1
                    
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
        
        # 更新进度条
        pbar.update(1)
    
    pbar.close()
    
    # 根据实际生成的问答对数量动态命名文件
    actual_count = len(all_qa_pairs)
    output_path = f"{OUTPUT_PATH_QA_STATEMENT}/{world}_{role}_qa_{actual_count}.json"
    
    # 检查同名文件是否存在，如果存在则添加时间戳避免覆盖
    if os.path.exists(output_path):
        timestamp = int(time.time())
        output_path = f"{OUTPUT_PATH_QA_STATEMENT}/{world}_{role}_qa_{actual_count}_{timestamp}.json"
    
    # 保存生成的问答对
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
    print(f"已生成 {actual_count} 个问答对，保存到 {output_path}")
