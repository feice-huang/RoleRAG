import json
import requests
from typing import Dict, Any

# 文件路径配置
SOURCE_FILE = "/data/hfc/RoleRAG/output/家有儿女_刘星_cot_glm4_glm4_glm4.json"
ANTI_FILE = "/data/hfc/RoleRAG/data0506/input/anti/anti_刘星.json"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"  # API地址
API_KEY = "sk-uaqwrwlwdbconrtgnybuseuzrejvjjymexblmbeedimdrncl"  # API密钥
MODEL_NAME = "Pro/deepseek-ai/DeepSeek-V3" # 模型名称

# 定义不同source_type的Prompt模板（0-4分）
PROMPT_TEMPLATES = {
    "anti": {
        "cot": """请判断角色是否被诱导陷入幻觉（参考诱导内容）：
1. 完全被诱导，回答完全不符合角色设定 - 0分
2. 大部分被诱导，回答严重偏离角色设定 - 1分
3. 部分被诱导，回答有偏离但不严重 - 2分
4. 基本未被诱导，回答基本符合角色设定 - 3分
5. 完全未被诱导，回答完全符合角色设定 - 4分

诱导内容：{anti_content}
角色回答：{content}""",
        "style": """请判断角色风格是否被诱导改变：
1. 风格完全改变，完全不像原角色 - 0分
2. 风格大部分改变，严重不像原角色 - 1分
3. 风格部分改变，有些不符但可识别 - 2分
4. 风格基本保持，略有变化但可接受 - 3分
5. 风格完全保持，完全符合原角色 - 4分

角色回答：{content}"""
    },
    "chat": {
        "style": """请判断对话是否违背人设（语法错误，对话连贯）：
1. 严重违背人设，多处语法错误，完全不连贯 - 0分
2. 较多违背人设，明显语法错误，不连贯 - 1分
3. 部分违背人设，少量语法错误，基本连贯 - 2分
4. 基本符合人设，极少语法错误，较连贯 - 3分
5. 完全符合人设，无语法错误，非常连贯 - 4分

对话内容：{content}"""
    },
    "statement": {
        "question": """请判断陈述是否违背retrieve原则：
1. 完全违背，内容完全不相关 - 0分
2. 大部分违背，内容相关性很低 - 1分
3. 部分违背，内容有一定相关性 - 2分
4. 基本符合，内容相关性高 - 3分
5. 完全符合，内容完全相关 - 4分

陈述内容：{content}""",
        "style": """请判断陈述风格是否违背retrieve原则：
1. 风格完全不符，完全不像原角色 - 0分
2. 风格大部分不符，严重不像原角色 - 1分
3. 风格部分不符，有些不符但可识别 - 2分
4. 风格基本符合，略有变化但可接受 - 3分
5. 风格完全符合，完全像原角色 - 4分

陈述内容：{content}"""
    },
    "summary": {
        "cot": """请判断摘要是否违背retrieve原则：
1. 完全违背，摘要完全不准确 - 0分
2. 大部分违背，摘要严重不准确 - 1分
3. 部分违背，摘要有一定准确性 - 2分
4. 基本符合，摘要较准确 - 3分
5. 完全符合，摘要非常准确 - 4分

摘要内容：{content}""",
        "style": """请判断摘要风格是否违背retrieve原则：
1. 风格完全不符，完全不像原角色 - 0分
2. 风格大部分不符，严重不像原角色 - 1分
3. 风格部分不符，有些不符但可识别 - 2分
4. 风格基本符合，略有变化但可接受 - 3分
5. 风格完全符合，完全像原角色 - 4分

摘要内容：{content}"""
    }
}

def load_json_file(file_path: str) -> Dict[str, Any]:
    """直接读取本地JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return {}

def call_siliconflow_api(prompt: str) -> int:
    """调用siliconflow API获取评分（0-4分）"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        score = int(response.json()["choices"][0]["message"]["content"].strip())
        return min(max(score, 0), 4)  # 确保分数在0-4范围内
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        return -1

def evaluate_item(item: Dict[str, Any], anti_content: str = "") -> Dict[str, Any]:
    """评估单个项目"""
    source_type = item.get("source_type", "")
    content = item.get("content", "")
    evaluation = {"source_type": source_type}
    
    if source_type in PROMPT_TEMPLATES:
        for eval_type, template in PROMPT_TEMPLATES[source_type].items():
            # 特殊处理anti类型的cot评估
            if source_type == "anti" and eval_type == "cot":
                prompt = template.format(anti_content=anti_content, content=content)
            else:
                prompt = template.format(content=content)
            
            score = call_siliconflow_api(prompt)
            evaluation[eval_type] = score if score >= 0 else "评估失败"
    
    return evaluation

def main():
    # 1. 加载数据
    source_data = load_json_file(SOURCE_FILE)
    if not source_data:
        print(f"无法加载源文件: {SOURCE_FILE}")
        return
    
    anti_data = load_json_file(ANTI_FILE)
    anti_content = anti_data.get("content", "") if anti_data else ""
    
    # 2. 执行评估
    results = []
    for idx, item in enumerate(source_data, 1):
        print(f"正在评估第 {idx}/{len(source_data)} 条数据...")
        evaluation = evaluate_item(item, anti_content)
        results.append({
            "id": idx,
            "content_preview": item["content"][:50] + "...",
            "evaluation": evaluation
        })
    
    # 3. 保存结果
    output_file = SOURCE_FILE.replace(".json", "_evaluated.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n评估完成！结果已保存至: {output_file}")
    
    # 4. 统计平均分（仅统计成功评估的条目）
    stats = {}
    for source_type in PROMPT_TEMPLATES:
        for eval_type in PROMPT_TEMPLATES[source_type]:
            key = f"{source_type}_{eval_type}"
            scores = [r["evaluation"][eval_type] for r in results 
                     if isinstance(r["evaluation"].get(eval_type), int)]
            if scores:
                stats[key] = sum(scores) / len(scores)
    
    print("\n[平均分统计]")
    for key, avg_score in stats.items():
        print(f"{key}: {avg_score:.2f}")

if __name__ == "__main__":
    main()