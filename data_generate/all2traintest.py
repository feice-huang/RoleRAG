import json
import os
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 需要改的
world = "家有儿女"
role = "刘星"
input_file = f"/data/hfc/RoleRAG/data0506/all/家有儿女_刘星_qa_203.json"


train_dir = "/data/hfc/RoleRAG/data0506/train"
test_dir = "/data/hfc/RoleRAG/data0506/test"

# 创建输出目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 读取JSON文件
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 按照source_type分组
grouped_data = defaultdict(list)
for item in data:
    source_type = item.get("source_type", "unknown")
    grouped_data[source_type].append(item)

# 分层采样
train_data = []
test_data = []
source_type_counts = {}

for source_type, items in grouped_data.items():
    train_items, test_items = train_test_split(items, test_size=0.2, random_state=42)
    train_data.extend(train_items)
    test_data.extend(test_items)
    source_type_counts[source_type] = {"train": len(train_items), "test": len(test_items)}

# 保存分层采样后的数据
train_output_file = os.path.join(train_dir, f"{world}_{role}_train.json")
test_output_file = os.path.join(test_dir, f"{world}_{role}_test.json")

with open(train_output_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(test_output_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

# 打印每种source_type的数量
print("分层采样后的数据分布：")
for source_type, counts in source_type_counts.items():
    print(f"Source Type: {source_type}, Train: {counts['train']}, Test: {counts['test']}")