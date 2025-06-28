# MODEL_PATH=$1
# MODEL_NAME=$2
# CONV_MODE=$3

# accelerate launch --num_processes=8\
#            evaluate_lmms_eval.py \
#            --model eagle \
#            --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
#            --tasks  mme,seed_bench,pope,scienceqa_img,gqa,ocrbench,textvqa_val,chartqa \
#            --batch_size 1 \
#            --log_samples \
#            --log_samples_suffix ${MODEL_NAME}_mmbench_mathvista_seedbench \
#            --output_path ./logs/ 

# MODEL_PATH=./model/Eagle-X5-7B
# MODEL_PATH=./model/Eagle-X4-8B-Plus
# MODEL_PATH=./checkpoints/final_result/image/finetune-eagle-x1-llama3.2-1b-image_L
# MODEL_PATH=./checkpoints/final_result/video/video_finetune_1epoch
# MODEL_PATH=./checkpoints/Baseline/Video/pr_llm/video_llama3.2-1b-finetune_pr_llm/checkpoint-2000
# MODEL_PATH=./checkpoints/Baseline/Audio/finetune/en_pr/finetune-audio-llama3.2-3b
# MODEL_PATH=./checkpoints/Baseline/Image/finetune/en_pr/finetune-eagle-x1-llama3.2-3b-image_L
# MODEL_PATH=./qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-doc-chart
# MODEL_PATH=./qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-doc-chart-before
MODEL_PATH=./qbs/Eagle_LanguageBind/checkpoints/disk2/Images/finetune/pr_llm/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-eagle/checkpoint-10
# MODEL_PATH=./checkpoints/final_result1/audio_finetune_1epoch
MODEL_NAME=eagle
# CONV_MODE=vicuna_v1
CONV_MODE=llama3
# TASKS=ocrbench_v2
TASKS=chartqa
# TASKS=docvqa,chartqa,textvqa_val
# TASKS=mme,seedbench,pope,scienceqa_img,gqa,ocrbench,textvqa_val,chartqa,docvqa
# TASKS=activitynetqa,egoschema,mvbench,perceptiontest_test_mc,videomme,videochatgpt,youcook2_val
# TASKS=mvbench
# TASKS=youcook2_val
# TASKS=librispeech
# TASKS=clotho_aqa_test

CUDA_VISIBLE_DEVICES='0' accelerate launch --num_processes=1 \
           evaluate_lmms_eval_fzy.py \
           --model eagle \
           --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
           --tasks ${TASKS} \
           --batch_size 1 \
           --log_samples \
           --log_samples_suffix ${MODEL_NAME}_mmbench \
           --output_path ./output/eval/ 
echo $MODEL_PATH
echo $TASKS
# CUDA_VISIBLE_DEVICES='1' python evaluate_lmms_eval.py \
#            --model eagle \
#            --model_args pretrained=${MODEL_PATH},conv_template=${CONV_MODE} \
#            --tasks ${TASKS} \
#            --batch_size 1 \
#            --log_samples \
#            --log_samples_suffix ${MODEL_NAME}_mmbench \
#            --output_path ./output/eval/ 
        #    --verbosity=DEBUG
# echo $MODEL_PATH
# echo $TASKS