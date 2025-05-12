import os
import json
import random
from glob import glob

QA_DIR = "/data/hfc/RoleRAG/data0506/qa"
OUTPUT_DIR = "/data/hfc/RoleRAG/data0506/all"

def qa2all(world, role):
    """
    合并并打乱指定 world 和 role 的所有 JSON 文件。
    :param world: 世界名称
    :param role: 角色名称
    """
    def load_and_filter_json(file_path, source_type):
        """
        加载 JSON 文件并提取需要的字段。
        :param file_path: JSON 文件路径
        :param source_type: 数据来源类型
        :return: 提取后的数据列表
        """
        hallucination_mapping = {
            "能力越界幻觉": "overreach",
            "能力不足幻觉": "underreach",
            "诱导性幻觉": "induction"
        }

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 只保留 "question", "answer", "retrieve" 字段，并根据条件调整 source_type
            return [
                {
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "retrieve": item.get("retrieve", ""),
                    "source_type": f"qa_anti_{hallucination_mapping.get(item.get('hallucination', '无'), 'unknown')}" 
                                   if source_type == "qa_anti" else source_type
                }
                for item in data
            ]
    
    # 定义需要遍历的子文件夹
    subfolders = ["qa_conv", "qa_anti", "qa_chat", "qa_statement", "qa_summary"]
    merged_data = []

    # 遍历子文件夹
    for subfolder in subfolders:
        folder_path = os.path.join(QA_DIR, subfolder)
        if not os.path.exists(folder_path):
            continue

        # 查找以 {world}_{role} 开头的 JSON 文件
        pattern = os.path.join(folder_path, f"{world}_{role}_*.json")
        for file_path in glob(pattern):
            print(f"Processing file: {file_path}")
            merged_data.extend(load_and_filter_json(file_path, subfolder))

    # 打乱数据
    random.shuffle(merged_data)

    # 保存合并后的数据
    count = len(merged_data)
    output_file = os.path.join(OUTPUT_DIR, f"{world}_{role}_qa_{count}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"Merged and shuffled data saved to: {output_file}")

if __name__ == "__main__":
    # 示例：合并 "家有儿女" 世界中的 "刘星" 角色
    qa2all("家有儿女", "刘星")