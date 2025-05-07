"""
wiki -> statement -> qa_statement(75) 
** conversations -> summary -> qa_summary(25) 写在qa_summary和qa_statement里面了
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




def generate_qa_for_scene(client, scene_data, character, model_name):
    """
    为单个场景生成问答对
    :param scene_data: 场景数据
    :param character: 角色名称
    :param model_name: OpenAI模型名称
    :return: 包含问题和回答的字典
    """
    scene_id = scene_data["scene_id"]
    summary = scene_data["summary"]
    
    # 构建提示
    prompt = f"""你正在扮演电视剧《家有儿女》中的"{character}"角色。请根据以下场景摘要，创建一个问题和回答对。
    
场景摘要：{summary}

要求：
1. 问题应该是对"{character}"角色本人在这个场景中行为、感受或想法的提问
2. 这段对话没有上下文，所以问题需要给出一些提示，让角色能够回想起当前讨论的是哪件事。不能简单地说“在这个场景中”类似的话
3. 回答应该站在"{character}"的角度，以第一人称语态回答，展现该角色的个性和情感
4. 回答中应包含场景相关的细节，且表现角色特点，不要虚构内容
5. 回答长度约30-50字

请按如下格式输出：
问题：[问题内容]
回答：[回答内容]"""

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 调用OpenAI模型
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        qa_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用API时出错: {e}")
        # 出错时返回一个简单的默认值
        qa_text = f"问题：你在这个场景中的感受如何？\n回答：作为{character}，我无法提供具体回答。(API调用失败)"
    
    # 解析问题和回答
    try:
        question_part = qa_text.split('问题：')[1].split('回答：')[0].strip()
        answer_part = qa_text.split('回答：')[1].strip()
    except:
        # 如果解析失败，使用整个输出
        question_part = "解析失败，请检查输出格式"
        answer_part = qa_text
    
    # 构建结果字典
    result = {
        "scene_id": scene_id,
        "character": character,
        "summary": summary,
        "question": question_part,
        "answer": answer_part,
        "original_scene": scene_data["scene_content"] if "scene_content" in scene_data else ""
    }
    
    return result

def summary2qa(world, character, apikey, model_name="gpt-4o"):
    """
    为特定角色的所有场景生成问答对
    :param scenes_file: 角色场景文件路径
    :param character: 角色名称
    :param model_name: OpenAI模型名称
    :param output_file: 输出文件路径
    """
    scenes_file = f"/data/hfc/RoleRAG/data0506/process/summary/{world}_{character}_summary.json"
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=apikey,
        base_url="https://api.chatanywhere.tech/v1"
        # base_url="https://api.chatanywhere.org/v1"
)
    def load_json(file_path):
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    
    # 加载场景数据
    scenes = load_json(scenes_file)
    print(f"共读取 {len(scenes)} 个'{character}'角色的场景数据")
    
    # 生成问答对
    qa_pairs = []
    for scene in tqdm(scenes, desc=f"为'{character}'生成问答对"):
        qa_pair = generate_qa_for_scene(client, scene, character, model_name)
        qa_pairs.append(qa_pair)
        
        # 打印进度信息
        print(f"场景 {scene['scene_id']} 问答对生成完成")
        print(f"摘要: {scene['summary'][:100]}...")
        print(f"问题: {qa_pair['question']}")
        print(f"回答: {qa_pair['answer']}")
        print("-" * 50)
        
        # 添加延迟以避免API速率限制
        time.sleep(1)
        
        # # 每处理几个场景保存一次中间结果
        # if len(qa_pairs) % 5 == 0:
        #     with open(output_file, "w", encoding="utf-8") as f:
        #         json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
        #     print(f"已保存中间结果，完成进度: {len(qa_pairs)}/{len(scenes)}")
    count = len(qa_pairs)
    output_dir = "/data/hfc/RoleRAG/data0506/qa/qa_summary"
    output_file = os.path.join(output_dir, f"{world}_{character}_qa_{count}.json")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # 保存最终结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=4)
    print(f"所有 {len(qa_pairs)} 个问答对已生成并保存至 {output_file}")
    
    return qa_pairs