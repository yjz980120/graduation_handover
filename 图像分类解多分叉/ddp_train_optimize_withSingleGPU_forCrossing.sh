#!/usr/bin/env sh

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : ddp.sh
#   Author       : Yufeng Liu
#   Date         : 2021-04-10
#   Description  : 
#
#================================================================

exp_folder="exps/exp004"
mkdir -p $exp_folder
#CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --deterministic --max_epochs 50 --save_folder ${exp_folder} --amp > ${exp_folder}/fullsize_adam.log &

filename=train-$(date +%Y_%m_%d)-$(date +%H_%M_%S)
#touch ${filename}.log

export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export NODE_RANK=0
export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

CUDA_VISIBLE_DEVICES=0 nohup \
python -u \
    train_crossing_threeUniteModality.py \
    > ${exp_folder}/${filename}.log & 

