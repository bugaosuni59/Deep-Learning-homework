#!/usr/bin/env bash
cd ..
CUDA_VISIBLE_DEVICES=1,6 python main.py \
    ntu RGB \
    --ada_type teda \
    --train_list data_list/NTU01_L.txt \
    --val_list data_list/NTU01_MR.txt \
    --num_class 60 \
    --arch resnet50 \
    --num_segments 3 \
    --consensus_type TEB \
    --batch-size 10 \
    --snapshot_pref ntu_tenet_lmr \
    --lr 0.001 \
    --gd 20
