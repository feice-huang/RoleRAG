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

from RoleRAG.data_generate.wiki2statement import load_json

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-uOftWQfqs2MwIAJfiwTPbqMFT8qAJqEWeWOFxC0MZVui10If",
    base_url="https://api.chatanywhere.tech/v1"
    # base_url="https://api.chatanywhere.org/v1"
)

def generate_scene_summary(scene_data, model_name):
    """
    为单个场景生成摘要
    :param scene_data: 场景数据
    :param model_name: OpenAI模型名称
    :return: 场景摘要
    """
    scene_id = scene_data["scene_id"]
    roles = scene_data["roles"]
    content = scene_data["scene_content"]
    
    # 构建提示
    prompt = f"""请为以下电视剧场景生成一个简洁的摘要，只保留核心情节和人物互动：

场景编号: {scene_id}
参与角色: {', '.join(roles)}
场景内容:
{content}

请用50-100字概括这个场景的核心情节，注意保留重要的关键信息、情感变化和人物动机，但不要有任何发散，尤其不要使用"揭示""暗示""体现"等词。注意不要用代指，每一处都直接使用人名。"""

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 调用OpenAI模型
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用API时出错: {e}")
        # 出错时返回一个简单的默认值
        summary = f"场景{scene_id}中，{', '.join(roles)}进行了互动。(API调用失败)"
    
    # 将原始数据与生成的摘要结合
    result = {
        "scene_id": scene_id,
        "roles": roles,
        "scene_content": content,
        "summary": summary.strip()
    }
    
    return result

def generate_all_summaries(scenes, model_name, output_path):
    """
    为所有场景生成摘要
    :param scenes: 场景数据列表
    :param model_name: OpenAI模型名称
    :param output_path: 输出文件路径
    :return: 带摘要的场景列表
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"共读取 {len(scenes)} 个场景数据")
    
    # 生成每个场景的摘要
    summarized_scenes = []
    for scene in tqdm(scenes, desc="生成场景摘要"):
        summarized_scene = generate_scene_summary(scene, model_name)
        summarized_scenes.append(summarized_scene)
        
        # 打印进度信息
        print(f"场景 {scene['scene_id']} 摘要生成完成")
        print(f"原始内容: {scene['scene_content'][:100]}...")  
        print(f"生成摘要: {summarized_scene['summary']}")
        print("-" * 50)
        
        # 添加延迟以避免API速率限制
        time.sleep(1)
        
        # 每处理几个场景保存一次中间结果，防止意外丢失
        if (scene['scene_id'] + 1) % 5 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summarized_scenes, f, ensure_ascii=False, indent=4)
            print(f"已保存中间结果，完成进度: {scene['scene_id'] + 1}/{len(scenes)}")
    
    # 保存最终结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summarized_scenes, f, ensure_ascii=False, indent=4)
    
    print(f"所有 {len(summarized_scenes)} 个场景摘要已生成并保存至 {output_path}")
    
    return summarized_scenes

def filter_scenes_by_role(scenes, role):
    """
    根据角色筛选场景
    :param scenes: 场景数据列表
    :param role: 角色名称
    :return: 包含指定角色的场景列表
    """
    filtered_scenes = []
    
    for scene in scenes:
        # 检查角色是否在场景的roles列表中
        if "roles" in scene and role in scene["roles"]:
            filtered_scenes.append(scene)
    
    return filtered_scenes

def save_scenes_for_roles(scenes, roles, output_dir):
    """
    为每个角色保存包含该角色的场景，只保留scene_id、roles和summary字段
    :param scenes: 场景数据列表
    :param roles: 角色列表
    :param output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for role in roles:
        # 筛选包含该角色的场景
        role_scenes_full = filter_scenes_by_role(scenes, role)
        
        # 只保留需要的三个字段
        role_scenes = []
        for scene in role_scenes_full:
            simplified_scene = {
                "scene_id": scene["scene_id"],
                "roles": scene["roles"],
                "summary": scene["summary"]
            }
            role_scenes.append(simplified_scene)
        
        output_path = os.path.join(output_dir, f"家有儿女_{role}_summary.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(role_scenes, f, ensure_ascii=False, indent=4)
        
        print(f"已为角色 {role} 保存 {len(role_scenes)} 个场景至 {output_path}，每个场景只保留scene_id、roles和summary字段")


# 直接设置参数
WORLD = "家有儿女"
MODEL_NAME = "gpt-4o-mini"  # 使用OpenAI的4o-mini模型

# 生成所有角色的思维链数据
ROLES = ["刘星", "刘梅", "夏东海", "小雨", "小雪"]

if __name__ == "__main__":
    # 读入场景数据
    scene_summary_path = "/data/hfc/datasets/RoleAgentBench/家有儿女 S1E1/raw/scene_summary.json"
    output_path = "/data/hfc/RoleRAG/mydata/summary/家有儿女_summary.json"
    output_dir = "/data/hfc/RoleRAG/mydata/summary"
    
    # 从现有文件加载已生成摘要的场景数据
    if os.path.exists(output_path):
        print(f"从现有文件加载已生成的摘要数据: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            summarized_scenes = json.load(f)
    else:
        # 加载原始场景数据
        scenes = load_json(scene_summary_path)
        # 为所有场景生成摘要
        summarized_scenes = generate_all_summaries(scenes, MODEL_NAME, output_path)
    
    # 为每个角色筛选场景并保存
    save_scenes_for_roles(summarized_scenes, ROLES, output_dir)
    
    print("所有角色的场景摘要已生成并保存完毕！")