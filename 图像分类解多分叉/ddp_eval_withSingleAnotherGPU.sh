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

exp_folder="exps/exp007/exp007_predictTest"
mkdir -p $exp_folder
#CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --deterministic --max_epochs 50 --save_folder ${exp_folder} --amp > ${exp_folder}/fullsize_adam.log &


export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export NODE_RANK=0
export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))
export MASTER_PORT=29501

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=1 \
python -u -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    --master_port=$MASTER_PORT \
    evaluate_withCldice.py \
    --save_folder ${exp_folder} \
    --deterministic \
    --amp \
    --image_shape 128,160,160 \
    --batch_size 1 \
    --evaluation \
    --checkpoint ${exp_folder}/../best_model.pt \
    --data_file ./data/task002_allData/data_splits.pkl
    

