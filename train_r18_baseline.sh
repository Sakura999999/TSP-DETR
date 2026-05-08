#!/bin/bash

# DAB-DETR + ResNet18 Baseline on DOTA (纯 baseline, 单模型)
export CUDA_VISIBLE_DEVICES=1

python -m torch.distributed.launch --use_env --nproc_per_node=1 \
    --master_port=29501 \
    main.py \
    --output_dir output/dab_detr_r18_dota_baseline_v2 \
    --dataset_file dota \
    --coco_path ./data/dota \
    --modelname dab_detr \
    --backbone resnet18 \
    --epochs 150 \
    --lr_drop 120 \
    --batch_size 20 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --num_workers 12 \
    --clip_max_norm 0.1 \
    --find_unused_params \
    --amp
