import json
from retriever import Retriever  # 假设上面的类保存在 retriever.py

# 初始化
retriever = Retriever(config={
    "embedding_model": "/data/hfc/checkpoints/text2vec-large-chinese"
})

# file_paths = [
#     f"/data/hfc/RoleRAG/mydata/input/wiki/wiki_刘星.txt",
#     f"/data/hfc/RoleRAG/mydata/input/wiki/wiki_家有儿女.txt",
#     "/data/hfc/datasets/RoleAgentBench/家有儿女 S1E1/profiles/刘星.jsonl"
# ]

# docs = retriever.load_files(file_paths)
# retriever.create_vector_store(docs)

# # 保存
# retriever.save_vector_store("/data/hfc/faiss_store/ragtest")
retriever.load_vector_store("/data/hfc/faiss_store/ragtest", allow_dangerous_deserialization=True)

# 检索
query = "螃蟹和蜘蛛"
retrieved_docs = retriever.retrieve_top_k(query, k=3, with_context=True)

for i, doc in enumerate(retrieved_docs, 1):
    print(f"=== 第 {i} 条检索结果 ===")
    
    # 注意，这里提取真正命中的小段内容
    target_text = doc.metadata.get("original_content", "").strip()
    if not target_text:
        target_text = doc.page_content.strip()

    context_lines = doc.page_content.split("\n")

    matched = False
    for line in context_lines:
        if target_text in line:
            print(f"【匹配】{line}")
            matched = True
        else:
            print(f"    {line}")
    
    if not matched:
        print("⚠️ 注意：没有找到命中的句子，高亮失败。")

    print(f"元数据: {doc.metadata}")
    print()


