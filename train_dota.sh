#!/bin/bash

# Set CUDA device (0是第一张卡，1是第二张卡)
export CUDA_VISIBLE_DEVICES=1

# Define common arguments
GPUS=1
BATCH_SIZE=10
EPOCHS=150
LR_DROP=80
MODEL_NAME="dab_detr"
TEACHER_BACKBONE="resnet101"  # Teacher: DAB-DETR + ResNet101 (pretrained, DC5)
STUDENT_BACKBONE="resnet50"   # Student: DAB-DETR + ResNet50
TEACHER_DILATION="--teacher_dilation"  # R101 pretrained weights use DC5 dilation
PRETRAIN_MODEL_PATH="pretrained/DAB_R101/DAB-R101-DC5.pth"  # Must be ResNet101 pretrained weights
LR=1e-5
LR_BACKBONE=1e-6
CLIP_MAX_NORM=0.1
NUM_WORKERS=12

# Knowledge Distillation flags
LOSS_KD_LOGITS=True        # 开启 logits 级 KD 蒸馏（KL散度分类 + SmoothL1回归）
AUX_REFPINTS="--aux_refpoints" # 开启 teacher anchor 蒸馏

# Dataset and output paths
DATASET_FILE="coco"   # Use 'coco' for 91-class to match pretrained teacher; DOTA labels (1-15) are subset
COCO_PATH="./data/dota"
OUTPUT_DIR="output/kd_dota"

# Optional: resume from a previous checkpoint (leave empty for fresh training)
RESUME="output/kd_dota/checkpoint.pth"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Command to run KD training
# 注意这里：我帮你加上了 --use_env，彻底解决 local-rank 报错问题！
python -m torch.distributed.launch --use_env --nproc_per_node=${GPUS} \
    --master_port=29500 \
    kd_main.py \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_file "${DATASET_FILE}" \
    --coco_path "${COCO_PATH}" \
    --modelname "${MODEL_NAME}" \
    --backbone "${STUDENT_BACKBONE}" \
    --teacher_backbone "${TEACHER_BACKBONE}" \
    ${TEACHER_DILATION} \
    --epochs "${EPOCHS}" \
    --lr_drop "${LR_DROP}" \
    --batch_size "${BATCH_SIZE}" \
    --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
    --lr "${LR}" \
    --lr_backbone "${LR_BACKBONE}" \
    --num_workers "${NUM_WORKERS}" \
    --clip_max_norm "${CLIP_MAX_NORM}" \
    --loss_kd_logits "${LOSS_KD_LOGITS}" \
    ${AUX_REFPINTS} \
    --find_unused_params \
    --amp \
    --resume "${RESUME}"