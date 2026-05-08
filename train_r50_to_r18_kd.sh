#!/bin/bash

# Knowledge Distillation: DAB-DETR R50 (Teacher) → DAB-DETR R18 (Student) on DOTA
export CUDA_VISIBLE_DEVICES=0

GPUS=1
BATCH_SIZE=8
EPOCHS=150
LR_DROP=120
LR=1e-4
LR_BACKBONE=1e-5

# Teacher: DAB-DETR + R50 (trained on DOTA, 20 classes)
TEACHER_BACKBONE="resnet50"
PRETRAIN_MODEL_PATH="pretrained/DOTA_R50/checkpoint.pth"

# Student: DAB-DETR + R18
STUDENT_BACKBONE="resnet18"

OUTPUT_DIR="output/dab_r50_to_r18_dota"
COCO_PATH="./data/dota"

mkdir -p "${OUTPUT_DIR}"

# 注意: --dataset_file coco → num_classes=91, 匹配 COCO 预训练的 teacher
# DOTA 的类别 ID (1-15) 是 91 类的子集，可以正常训练
python -m torch.distributed.launch --use_env --nproc_per_node=${GPUS} \
    --master_port=29500 \
    kd_main.py \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_file coco \
    --coco_path "${COCO_PATH}" \
    --modelname dab_detr \
    --backbone "${STUDENT_BACKBONE}" \
    --teacher_backbone "${TEACHER_BACKBONE}" \
    --epochs "${EPOCHS}" \
    --lr_drop "${LR_DROP}" \
    --batch_size "${BATCH_SIZE}" \
    --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
    --lr "${LR}" \
    --lr_backbone "${LR_BACKBONE}" \
    --num_workers 12 \
    --clip_max_norm 0.1 \
    --loss_kd_logits True \
    --aux_refpoints \
    --random_refpoints 300 \
    --find_unused_params \
    --resume "${OUTPUT_DIR}/checkpoint.pth"
