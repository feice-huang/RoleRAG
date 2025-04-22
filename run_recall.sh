# conda activate llama
cd /data/hfc/RoleRAG
# 下面CUDA_VISIBLE_DEVICES= 的卡数需要和yaml文件中的ray_num_workers一致
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 USE_RAY=1 llamafactory-cli train /data/hfc/RoleRAG/configs/train_lora/llama3_lora_sft_ray_recall.yaml

llamafactory-cli export configs/merge_lora/llama3_lora_sft_recall.yaml