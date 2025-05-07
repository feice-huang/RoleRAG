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
from glob import glob

from openai import OpenAI





QA_DIR = "/data/hfc/RoleRAG/data0506/qa"
COT_DIR = "/data/hfc/RoleRAG/data0506/process/cot"



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

def generate_cot(world, role, apikey, model_engine):
    """
    从 qa 数据生成 CoT（思考过程）
    """
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=apikey,
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.org/v1"
)
    qa_path = f"{COT_DIR}/{role}_temp_combined_qa.json"  # 修正路径为 JSON 文件
    # cot_path 将在生成数据后动态设置
    if not os.path.exists(qa_path):
        print(f"{qa_path} 不存在，无法生成 CoT。")
        return

    # 加载合并后的 QA 数据
    with open(qa_path, "r", encoding="utf-8") as f:
        dpo_data = json.load(f)

    count = len(dpo_data)
    cot_path = f"{COT_DIR}/{role}_cot_{count}.json"

    if os.path.exists(cot_path):
        print(f"{cot_path} 已存在，跳过生成步骤。")
        return

    os.makedirs(os.path.dirname(cot_path), exist_ok=True)

    cot_data = []

    for item in tqdm(dpo_data, desc="Generating CoT"):
        cot_prompt = fill_in_cot_template(world, role, item["question"], item["answer"], item.get("retrieve", ""))
        messages = [
            {"role": "user", "content": cot_history_input()},
            {"role": "assistant", "content": cot_history_output()},
            {"role": "user", "content": cot_prompt}
        ]

        # 使用 gpt-4o-mini 模型生成思考过程
        response = client.chat.completions.create(
            model=model_engine,
            messages=messages,
            temperature=1.0
        )
        generated_cot = response.choices[0].message.content.strip()

        print("messages:", messages)
        print("*******************")
        print("generated_cot:", generated_cot)

        cot_data.append({
            "system": "你是一个角色扮演专家，请以【问题重述】【实体确认】【逻辑推理】【分析回答】【最终回答】的顺序，以扮演角色的身份回答问题",
            "instruction": "",
            "input": get_cot_prompt(item["question"], item.get("retrieve", "")),
            "output": item["answer"],
            "cot": generated_cot
        })

    # 保存生成的 CoT 数据为 JSON 文件
    with open(cot_path, "w", encoding="utf-8") as f:
        json.dump(cot_data, f, ensure_ascii=False, indent=4)

    print(f"Saved CoT data to {cot_path}")


def qa2cot(world, role, apikey, model_engine="gpt-4o"):
    """
    从指定文件夹加载数据，合并、打乱顺序后生成 CoT
    """
    qa_folders = ["qa_chat", "qa_statement", "qa_summary"]
    qa_files = []

    # 遍历文件夹，收集符合条件的文件路径
    for folder in qa_folders:
        folder_path = os.path.join(QA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"文件夹 {folder_path} 不存在，跳过。")
            continue

        for file in glob(os.path.join(folder_path, f"{world}_{role}_*.json")):
            qa_files.append(file)

    if not qa_files:
        print(f"未找到任何匹配 {world}_{role} 的文件。")
        return

    # 合并所有文件内容
    combined_data = []
    for file in qa_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            combined_data.extend(data)

    # 打乱数据顺序
    random.shuffle(combined_data)

    # 保存合并后的数据到 JSON 文件
    temp_qa_path = f"{COT_DIR}/{role}_temp_combined_qa.json"
    os.makedirs(os.path.dirname(temp_qa_path), exist_ok=True)
    with open(temp_qa_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

    # 调用 generate_cot 生成 CoT
    generate_cot(world, role, apikey, model_engine)
    # 删除中间文件
    if os.path.exists(temp_qa_path):
        os.remove(temp_qa_path)
        print(f"已删除中间文件: {temp_qa_path}")


# 示例调用
if __name__ == "__main__":
    world = "家有儿女"
    role = "刘星"
    model_engine = "gpt-4o"
    apikey = "sk-MA7hKS37UdRUmP3Xz4BzHt3Rqj6QFbRoEagxcmFwwBBHyZR6"
    qa2cot(world, role, apikey, model_engine)