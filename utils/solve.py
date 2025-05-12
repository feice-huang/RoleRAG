import json

def filter_questions_and_answers(input_file, output_file_pre):
    """
    读取 JSON 文件，只保留每条数据的 "question" 和 "answer" 字段。
    :param input_file: 输入 JSON 文件路径
    :param output_file: 输出 JSON 文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 只保留 "question" 和 "answer" 字段
    filtered_data = [
        {
            "question": item["qa_pair"]["question"],
            "answer": item["qa_pair"]["answer"]
        }
        for item in data
    ]

    output_file = f"{output_file_pre}_{len(filtered_data)}.json"
    # 保存过滤后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)

    print(f"过滤后的数据已保存到: {output_file}")

# 示例用法
input_file = "/data/hfc/RoleRAG/data0506/process/conv/家有儿女_刘星_summary.json"
output_file_pre = "/data/hfc/RoleRAG/data0506/qa/qa_conv/家有儿女_刘星_qa"
filter_questions_and_answers(input_file, output_file_pre)