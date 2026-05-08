#!/bin/bash

# DAB-DETR + ResNet101 baseline on DOTA
# 不启用 DC5 dilation → stride=32 → encoder 特征图缩小 4 倍 → 显存安全
# Backbone 自动加载 ImageNet 预训练 (backbone.py 内置 pretrained=True)

BATCH_SIZE=10         
EPOCHS=150
LR_DROP=120
LR=1e-4
LR_BACKBONE=1e-5
NUM_QUERIES=600       # DOTA 密集场景

OUTPUT_DIR="output/dab_r101_dota"
COCO_PATH="./data/dota"

mkdir -p "${OUTPUT_DIR}"

python main.py \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_file coco \
    --coco_path "${COCO_PATH}" \
    --modelname dab_detr \
    --device cuda:1 \
    --backbone resnet101 \
    --epochs "${EPOCHS}" \
    --lr_drop "${LR_DROP}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lr_backbone "${LR_BACKBONE}" \
    --num_queries "${NUM_QUERIES}" \
    --num_select "${NUM_QUERIES}" \
    --num_workers 12 \
    --clip_max_norm 0.1
