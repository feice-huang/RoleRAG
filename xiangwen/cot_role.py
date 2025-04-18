import transformers
import torch
import re

# 加载模型
model_id = "/data3/dxw/data/sft_role_megred/"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# 提示模板
COT_PREFIX = "你是@@刘星@@，需要站在刘星角度进行COT3步骤推理，第1步理解问题中的人事物，第2步分析他们的时间地点关系，第3步在[]中给出结论："
STYLE_PREFIX = "你是@@刘星@@，需要用他的语气、风格来改写输入句子："

# 推理函数
def query_llm(prompt):
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        prompt,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
    )
    return outputs[0]["generated_text"].split("Output:")[-1].strip()

# 提取[]中内容
def extract_bracket_text(text):
    match = re.search(r"\[([^\[\]]+)\]", text)
    return match.group(1).strip() if match else None

# 主循环
print("=== 刘星两阶段语气回答系统 ===")
print("输入示例：小雨：哥，你为什么突然想交女朋友？")
print("输入 q 或 exit 退出。\n")

while True:
    user_input = input("请输入问题 > ").strip()
    if user_input.lower() in {"q", "exit"}:
        break

    # 阶段1：COT 推理
    cot_prompt = f"{COT_PREFIX}{user_input}\nOutput:"
    print("\n[阶段1] 生成COT推理中...\n")
    cot_output = query_llm(cot_prompt)
    print("【COT输出】：", cot_output)

    conclusion = extract_bracket_text(cot_output)
    if not conclusion:
        print("❌ 无法从中提取[]结论，请重试或检查模型输出。")
        continue

    # 阶段2：语气改写
    style_prompt = f"{STYLE_PREFIX}{conclusion}\nOutput:"
    print("\n[阶段2] 生成刘星语气改写中...\n")
    style_output = query_llm(style_prompt)

    print("【刘星风格回答】：", style_output)
    print("\n" + "=" * 50 + "\n")
