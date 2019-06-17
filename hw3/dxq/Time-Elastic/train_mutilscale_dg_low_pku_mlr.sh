#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6,7 python main.py \
    something RGB \
    --ada_type type_two \
    --train_list data_list/pku_M.txt \
    --val_list data_list/pku_LR.txt \
    --num_class 41 \
    --arch resnet50 \
    --num_segments 3 \
    --consensus_type TEB \
    --batch-size 10 \
    --snapshot_pref mutilscale_dg \
    --lr 0.001 \
    --gd 20
