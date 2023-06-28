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

exp_folder="exps/exp009/exp009_predictComplexSoma"
mkdir -p $exp_folder
#CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --deterministic --max_epochs 50 --save_folder ${exp_folder} --amp > ${exp_folder}/fullsize_adam.log &

filename=evaluate_$(date +%Y_%m_%d)_$(date +%H_%M_%S)

## 当有模型在训练的时候，经常没法同时去做预测，因为会报错说端口已经被占用了，在外面export MASTER_PORT指定别的端口却没有用，所以直接在代码里面设置端口，用torch.distributed.init_process_group()这个函数，就可以做到同时去预测了
## 代码如下：
# dist_init_method = 'tcp://{master_ip}:{master_port}'.format(master_ip='1    27.0.0.1', master_port='10000')
# dist_world_size = 1    #total number of distributed processes.
# distrib.init_process_group(backend="nccl", init_method=dist_init_method,     world_size=dist_world_size, rank=0)
export NUM_NODES=1
export NUM_GPUS_PER_NODE=1
export NODE_RANK=0
export WORLD_SIZE=$((NUM_NODES * $NUM_GPUS_PER_NODE))
#export MASTER_PORT=29501


# launch our script w/ `torch.distributed.launch`
CUDA_VISIBLE_DEVICES=0 nohup \
python -u -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    evaluate_forSoma_new_fixBugs.py \
    --save_folder ${exp_folder} \
    --deterministic \
    --amp \
    --image_shape 128,128,128 \
    --batch_size 1 \
    --evaluation \
    --checkpoint ${exp_folder}/../best_model.pt \
    --data_file /home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/exps/exp009/exp009_predictComplexSoma/data_splits.pkl \
    > ${exp_folder}/${filename}.log &
    

