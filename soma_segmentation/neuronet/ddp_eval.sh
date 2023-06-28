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

exp_folder="exps/exp008/exp008_predict_multiSoma_complex_onlyOne_forTestCodes"
mkdir -p $exp_folder
#CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --deterministic --max_epochs 50 --save_folder ${exp_folder} --amp > ${exp_folder}/fullsize_adam.log &


export NUM_NODES=1
export NUM_GPUS_PER_NODE=2
export NODE_RANK=0
export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))

# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=0,1 \
python -u -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    evaluate_forSoma.py \
    --save_folder ${exp_folder} \
    --deterministic \
    --amp \
    --image_shape 128,128,128 \
    --batch_size 1 \
    --evaluation \
    --checkpoint ${exp_folder}/../best_model.pt \
    --data_file ./data/task006_withMultiSomaOfflineAug_removeSomaBadMultiSoma/multiSoma_complex_onlyOne_forTestCodes/data_splits.pkl
    

