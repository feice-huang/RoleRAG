import json
import os
from tqdm import tqdm
from utils.solve import filter_questions_and_answers
def simple_test(role, profile_path, qa_path):
    """
    生成角色扮演的提示信息。
    :param role: 角色名称
    :param profile: 角色信息
    :param question: 提问内容
    :param observation: 参考信息
    :return: 格式化的提示字符串
    """
    # 读入/data/hfc/RoleRAG/data0506/input/general/general_刘星.txt的内容作为profile
    profile_path = f"/data/hfc/RoleRAG/data0506/input/general/general_{role}.txt"
    if not os.path.exists(profile_path):
        raise FileNotFoundError(f"角色信息文件不存在: {profile_path}")

    with open(profile_path, 'r', encoding='utf-8') as f:
        profile = f.read().strip()

    # 读入/data/hfc/RoleRAG/data0506/test/家有儿女_刘星_test.json的qa
    qa_path = f"/data/hfc/RoleRAG/data0506/test/{role}_test.json"
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"问答文件不存在: {qa_path}")
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        if not qa_data:
            raise ValueError(f"问答文件为空: {qa_path}")
        
    for qa_pair in qa_data:
        question = qa_pair.get("question")
        observation = qa_pair.get("retrieve", "")
        if not question:
            raise ValueError(f"问答对缺少问题: {qa_pair}")
        
        
        """ 然后就是用下面的prompt测试GPT了，
        1. Zero shot就是直接问GPT（observation留空）
        2. Few shot就是给一个示例（observation留空），比如可以
            messages = [
                {"role": "user", "content": history_input},
                {"role": "assistant", "content": history_output},
                {"role": "user", "content":prompt}
            ]
            这里的history_input和history_output就是写一个示例，注意这里的input也要套prompt模板
        3. RAG就是在Few shot的基础上，把这里的observation填上
        """

        # 生成提示信息
        prompt = f"""
    根据下面的角色扮演信息，回答问题。注意只需要给出答案，不需要任何解释和分析。
    你需要扮演的角色是：
    {role}
    你需要扮演的角色的信息：
    {profile}
    你需要回答的问题是：
    {question}
    已有的参考信息：
    {observation}

    """

    