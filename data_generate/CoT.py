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

from RoleRAG.data_generate.wiki2statement import call_tsinghua_deepseek, load_json, save_jsonl

def generate_scene_summary(scene_data, model_engine, token):
    """
    为单个场景生成摘要
    :param scene_data: 场景数据
    :param model_engine: 模型名称
    :param token: 授权令牌
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

请用50-100字概括这个场景的核心情节，注意保留重要的情感变化和人物动机，但不要有任何发散，尤其不要使用“揭示”“暗示”“体现”等词。注意不要用代指，每一处都直接使用人名。"""

    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 调用模型
    _, summary = call_tsinghua_deepseek(model_engine, token, messages)
    
    # 将原始数据与生成的摘要结合
    result = {
        "scene_id": scene_id,
        "roles": roles,
        "scene_content": content,
        "summary": summary.strip()
    }
    
    return result

# 直接设置参数
WORLD = "家有儿女"
MODEL_ENGINE = "DeepSeek-R1-671B"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjb2RlIjoiMTAyNyIsImlhdCI6MTc0NjQxNTA1NSwiZXhwIjoxNzQ2NDM2NjU1fQ.H9TzZVLg0mkUPoZMsx5yV0HSaui5yEiBoyIJidV5wK0"

# 生成所有角色的思维链数据
ROLES = ["刘星", "刘梅", "夏东海", "小雨", "小雪"]

if __name__ == "__main__":
    # 读入场景数据
    scene_summary_path = "/data/hfc/datasets/RoleAgentBench/家有儿女 S1E1/raw/scene_summary.json"
    output_path = "/data/hfc/RoleRAG/mydata/家有儿女_summary.json"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 加载原始场景数据
    scenes = load_json(scene_summary_path)
    print(f"共读取 {len(scenes)} 个场景数据")
    
    # 生成每个场景的摘要
    summarized_scenes = []
    for scene in tqdm(scenes, desc="生成场景摘要"):
        summarized_scene = generate_scene_summary(scene, MODEL_ENGINE, TOKEN)
        summarized_scenes.append(summarized_scene)
        
        # 打印进度信息
        print(f"场景 {scene['scene_id']} 摘要生成完成")
        print(f"原始内容: {scene['scene_content'][:100]}...")  
        print(f"生成摘要: {summarized_scene['summary']}")
        print("-" * 50)
        
        # 每处理几个场景保存一次中间结果，防止意外丢失
        if (scene['scene_id'] + 1) % 5 == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summarized_scenes, f, ensure_ascii=False, indent=4)
            print(f"已保存中间结果，完成进度: {scene['scene_id'] + 1}/{len(scenes)}")
    
    # 保存最终结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summarized_scenes, f, ensure_ascii=False, indent=4)
    
    print(f"所有 {len(summarized_scenes)} 个场景摘要已生成并保存至 {output_path}")