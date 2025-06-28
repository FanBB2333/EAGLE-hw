MODEL_PATH=/home1/hxl/disk2/MLLM/Models/VideoLLaMA3-7B
MODEL_PATH_2=/home1/hxl/disk2/MLLM/Models/VideoLLaMA3-7B-Image
CUDA_DEVICES=0,1
# HF_DATASETS_CACHE=/home1/hxl/disk/EAGLE/.cache/huggingface/datasets CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 ./lmms_eval/__main__.py \
#     --model videollama3 \
#     --model_args pretrained=${MODEL_PATH_2},use_flash_attention_2=True,device_map=auto \
#     --tasks mme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix VideoLLaMA3-7B_mme \
#     --output_path ./logs/

HF_DATASETS_CACHE=/home1/hxl/disk/EAGLE/.cache/huggingface/datasets CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 evaluate_lmms_eval.py \
    --model videollama3 \
    --model_args pretrained=${MODEL_PATH},use_flash_attention_2=True,device_map=auto \
    --tasks activitynetqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix VideoLLaMA3-7B_mme \
    --output_path ./logs/

# HF_DATASETS_CACHE=/home1/hxl/disk/EAGLE/.cache/huggingface/datasets CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python3 ./lmms_eval/__main__.py \
#     --model videollama3 \
#     --model_args pretrained=${MODEL_PATH_2},use_flash_attention_2=True,device_map=auto \
#     --tasks mvbench \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix VideoLLaMA3-7B_mme \
#     --output_path ./logs/