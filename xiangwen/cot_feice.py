import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

class SimpleInferenceEngine:
    def __init__(self, model_path: str, adapter_path: str = None, device: str = "cuda"):
        """
        初始化推理引擎，加载模型和适配器。
        :param model_path: 本地模型路径
        :param adapter_path: 本地适配器路径（可选）
        :param device: 使用的设备（如 "cuda" 或 "cpu"）
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        # 如果提供了适配器路径，加载适配器
        if adapter_path:
            self._load_adapter(adapter_path)

    def _load_adapter(self, adapter_path: str):
        """
        加载适配器到模型中。
        :param adapter_path: 本地适配器路径
        """
        adapter_state_dict = torch.load(adapter_path, map_location=self.device)
        self.model.load_state_dict(adapter_state_dict, strict=False)
        print(f"Adapter loaded from {adapter_path}")

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        使用模型生成文本。
        :param prompt: 输入的提示文本
        :param max_new_tokens: 最大生成的 token 数
        :param temperature: 生成的温度参数
        :param top_p: 核采样的概率阈值
        :return: 生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        outputs = self.model.generate(**inputs, generation_config=generation_config)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # 本地模型和适配器路径
    model_path = "/data/hfc/checkpoints/Llama-3.1-8B-Instruct"
    adapter_path = "/data/hfc/RoleRAG/saves/刘星_cot_800/llama3_8b_sft_lora/TorchTrainer_8d771_00000_0_2025-04-17_18-20-14/checkpoint_000003/checkpoint"  # 如果没有适配器，可以设置为 None

    # 初始化推理引擎
    engine = SimpleInferenceEngine(model_path, adapter_path)

    # 示例输入
    prompt = "你正在扮演刘星，请回答以下问题：\n问题：什么是人工智能？\n回答："
    response = engine.generate(prompt, max_new_tokens=100)
    print("生成的回答：", response)