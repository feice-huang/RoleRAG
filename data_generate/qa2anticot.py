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
COT_DIR = "/data/hfc/RoleRAG/data0506/process/anti_cot"



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
            {"role": "user", "content": anti_cot_history_input()},
            {"role": "assistant", "content": anti_cot_history_output()},
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
            "cot": generated_cot,
            "hallucination": item["hallucination"]
        })

    # 保存生成的 CoT 数据为 JSON 文件
    with open(cot_path, "w", encoding="utf-8") as f:
        json.dump(cot_data, f, ensure_ascii=False, indent=4)

    print(f"Saved CoT data to {cot_path}")


def qa2anticot(world, role, apikey, model_engine="gpt-4o"):
    """
    从指定文件夹加载数据，合并、打乱顺序后生成 CoT
    """
    qa_folders = ["qa_anti"]
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
    qa2anticot(world, role, apikey, model_engine)