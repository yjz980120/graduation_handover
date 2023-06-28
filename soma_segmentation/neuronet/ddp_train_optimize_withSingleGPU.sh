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

exp_folder="exps/exp014"
mkdir -p $exp_folder
#CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --deterministic --max_epochs 50 --save_folder ${exp_folder} --amp > ${exp_folder}/fullsize_adam.log &

filename=train-$(date +%Y_%m_%d)-$(date +%H_%M_%S)
#touch ${filename}.log

export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export NODE_RANK=0
export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=0 nohup \
python -u -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    train_forSoma.py \
    --deterministic \
    --max_epochs 70 \
    --save_folder ${exp_folder} \
    --amp \
    --use_robust_loss \
    --step_per_epoch 200 \
    --test_frequency 3 \
    --image_shape 128,128,128 \
    --batch_size 1 \
    --lr_steps '20,30,40,45,50,55,58,60,62,64,65' \
    --data_file ./data/task011_withoutMultiSomaOfflineAug_removeSomaBadMultiSoma_withShiftAug_edge_reweight/data_splits.pkl \
    > ${exp_folder}/${filename}.log & 

