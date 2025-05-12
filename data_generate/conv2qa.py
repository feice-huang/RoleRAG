import os
import json
import time
from tqdm import tqdm
from openai import OpenAI



def conv2qa(world, roles, api_key, model_name="gpt-4o-mini"):
    """
    封装整个文件功能为一个函数。
    :param world: 世界名称（如"家有儿女"）
    :param roles: 角色列表
    :param api_key: OpenAI API 密钥
    :param model_name: OpenAI 模型名称
    """
    key_role = ROLES[0]

    def load_json(file_path):
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.chatanywhere.tech/v1"
    )

    def generate_style_question(scene_data, key_role):
        """
        为单个场景生成问答对
        """
        scene_id = scene_data["scene_id"]
        roles = scene_data["roles"]
        content = scene_data["scene_content"]

        prompt = f"""我正在进行角色扮演，你需要帮助我生成数据集。请为以下电视剧场景生成一个问答对，要求：
1. 问答对的回答方是{key_role}，这个回答的具体内容需要尽可能符合{key_role}的说话语气，可以对场景中{key_role}的台词进行少量修改作为回答。
2. 问答对的提问方是用户。提问的问题需要和回答内容相配合。
3. 输出格式使用json，包含两个字段：question和answer，不要有多余输出。
【电视剧场景】
场景编号: {scene_id}
参与角色: {', '.join(roles)}
场景内容:
{content}
【示例输出】
{{
    "question": "",
    "answer": ""
}}
"""

        messages = [{"role": "user", "content": prompt}]
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            qa_pair = json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"调用API时出错: {e}")
            qa_pair = {"question": "API调用失败", "answer": "API调用失败"}

        return {
            "scene_id": scene_id,
            "roles": roles,
            "scene_content": content,
            "qa_pair": qa_pair
        }

    def generate_all_qa_pairs(scenes, output_path, key_role):
        """
        为所有场景生成问答对
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summarized_scenes = []
        for scene in tqdm(scenes, desc="生成场景QA for style"):
            summarized_scene = generate_style_question(scene, key_role)
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
                    "question": scene["qa_pair"]["question"],
                    "answer": scene["qa_pair"]["answer"],
                }
                for scene in filter_scenes_by_role(scenes, role)
            ]
            output_path = os.path.join(output_dir, f"{world}_{role}_conv.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(role_scenes, f, ensure_ascii=False, indent=4)
            print(f"已为角色 {role} 保存 {len(role_scenes)} 个场景至 {output_path}")

    # 文件路径配置
    scene_conv_path = f"/data/hfc/datasets/RoleAgentBench/{world} S1E1/raw/scene_summary.json"
    output_path = f"/data/hfc/RoleRAG/data0506/process/conv/{world}_conv.json"
    output_dir = f"/data/hfc/RoleRAG/data0506/process/conv"

    # 加载场景数据
    if os.path.exists(output_path):
        print(f"从现有文件加载已生成的问答数据: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            summarized_scenes = json.load(f)
    else:
        scenes = load_json(scene_conv_path)
        summarized_scenes = generate_all_qa_pairs(scenes, output_path, key_role)

    # 保存每个角色的场景摘要
    save_scenes_for_roles(summarized_scenes, roles, output_dir)
    # 删除中间文件
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"已删除中间文件: {output_path}")
    print("所有角色的场景摘要已生成并保存完毕！")

if __name__ == "__main__":
    WORLD = "家有儿女"
    ROLES = ["刘星"]
    API_KEY = "sk-uOftWQfqs2MwIAJfiwTPbqMFT8qAJqEWeWOFxC0MZVui10If"

    conv2qa(WORLD, ROLES, API_KEY)