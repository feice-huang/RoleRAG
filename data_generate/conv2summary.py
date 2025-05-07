import os
import json
import time
from tqdm import tqdm
from openai import OpenAI



def conv2summary(world, roles, api_key, model_name="gpt-4o-mini"):
    """
    封装整个文件功能为一个函数。
    :param world: 世界名称（如"家有儿女"）
    :param roles: 角色列表
    :param api_key: OpenAI API 密钥
    :param model_name: OpenAI 模型名称
    """
    def load_json(file_path):
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.chatanywhere.tech/v1"
    )

    def generate_scene_summary(scene_data):
        """
        为单个场景生成摘要
        """
        scene_id = scene_data["scene_id"]
        roles = scene_data["roles"]
        content = scene_data["scene_content"]

        prompt = f"""请为以下电视剧场景生成一个简洁的摘要，只保留核心情节和人物互动：

场景编号: {scene_id}
参与角色: {', '.join(roles)}
场景内容:
{content}

请用50-100字概括这个场景的核心情节，注意保留重要的关键信息、情感变化和人物动机，但不要有任何发散，尤其不要使用"揭示""暗示""体现"等词。注意不要用代指，每一处都直接使用人名。"""

        messages = [{"role": "user", "content": prompt}]
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
            summary = f"场景{scene_id}中，{', '.join(roles)}进行了互动。(API调用失败)"

        return {
            "scene_id": scene_id,
            "roles": roles,
            "scene_content": content,
            "summary": summary.strip()
        }

    def generate_all_summaries(scenes, output_path):
        """
        为所有场景生成摘要
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summarized_scenes = []
        for scene in tqdm(scenes, desc="生成场景摘要"):
            summarized_scene = generate_scene_summary(scene)
            summarized_scenes.append(summarized_scene)
            time.sleep(1)  # 避免API速率限制

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summarized_scenes, f, ensure_ascii=False, indent=4)
        return summarized_scenes

    def filter_scenes_by_role(scenes, role):
        """
        根据角色筛选场景
        """
        return [scene for scene in scenes if "roles" in scene and role in scene["roles"]]

    def save_scenes_for_roles(scenes, roles, output_dir):
        """
        为每个角色保存包含该角色的场景
        """
        os.makedirs(output_dir, exist_ok=True)
        for role in roles:
            role_scenes = [
                {
                    "scene_id": scene["scene_id"],
                    "roles": scene["roles"],
                    "summary": scene["summary"]
                }
                for scene in filter_scenes_by_role(scenes, role)
            ]
            output_path = os.path.join(output_dir, f"{world}_{role}_summary.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(role_scenes, f, ensure_ascii=False, indent=4)
            print(f"已为角色 {role} 保存 {len(role_scenes)} 个场景至 {output_path}")

    # 文件路径配置
    scene_summary_path = f"/data/hfc/datasets/RoleAgentBench/{world} S1E1/raw/scene_summary.json"
    output_path = f"/data/hfc/RoleRAG/data0506/process/summary/{world}_summary.json"
    output_dir = f"/data/hfc/RoleRAG/data0506/process/summary"

    # 加载场景数据
    if os.path.exists(output_path):
        print(f"从现有文件加载已生成的摘要数据: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            summarized_scenes = json.load(f)
    else:
        scenes = load_json(scene_summary_path)
        summarized_scenes = generate_all_summaries(scenes, output_path)

    # 保存每个角色的场景摘要
    save_scenes_for_roles(summarized_scenes, roles, output_dir)
    print("所有角色的场景摘要已生成并保存完毕！")

if __name__ == "__main__":
    WORLD = "家有儿女"
    ROLES = ["刘星", "刘梅", "夏东海", "小雨", "小雪"]
    API_KEY = "your_openai_api_key"

    conv2summary(WORLD, ROLES, API_KEY)