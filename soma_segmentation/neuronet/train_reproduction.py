#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Jingzhou Yuan (Braintell, Southeast University). All rights reserved.
#
#   Filename     : train.py
#   Author       : Jingzhou Yuan
#   Date         : 2021-05-14
#   Description  :
#
#================================================================

import os
import sys
import argparse
import numpy as np
import time
import json
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.utils.data as tudata
from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from neuronet.models import unet
from neuronet.utils import util
from neuronet.utils.image_util import unnormalize_normal
from neuronet.datasets.generic_dataset import GenericDataset
from neuronet.loss.dice_loss import BinaryDiceLoss

import path_util

parser = argparse.ArgumentParser(
    description="Segmentator for Neuronal Image With Pytorch")
## data specific
parser.add_argument("--data_file",default="./data/task001_soma/data_splits.pkl",                    type=str,help="dataset split file")
parser.add_argument("--batch_size",default=2,type=int,
                    help="Number of workers used in dataloading")
parser.add_argument("--image_shape",default="256,512,512",type=str,
                    help="Input image shape")
parser.add_argument("--cpu",action="store_true",
                    help="Whether to use gpu to train model,default True")
parser.add_argument("--amp",action="store_true",
                    help="Whether to use AMP training,default True"
parser.add_argument("-lr","--learning_rate",default=1e-2,type=float,
                    help="initial learning rate")
parser.add_argument("--momentum",default=0.99,type=float,
                    help="Momentum value for optim")
parser.add_argument("--weight_decay",default=3e-5,type=float,
                    help="Weight decay for SGD")
parser.add_argument("--use_robust_loss",action="store_true",
                    help="Whether to use robust loss")
parser.add_argument("--max_epochs",default=200,type=int,
                    help="Maximal number of epochs")
parser.add_argument("--step_per_epoch",default=200,type=int,
                    help="step per epoch")
parser.add_argument("--deterministic",action="store_true",
                    help="run in deterministic mode")
parser.add_argument("--test_frequency",default=3,type=int,
                    help="frequency of test")
parser.add_argument("--print_frequency",default=5,type=int,
                    help="frequency of information logging")
parser.add_argument("--local_rank",default=-1,type=int,metavar="N",
                    help="Local process rank") ## DDP required
parser.add_argument("--seed",default=1025,type=int,
                    help="Random seed value")
parser.add_argument("--checkpoint",default="",type=str,
                    help="Saved checkpoint")
parser.add_argument("--evaluation",action="store_true",
                    help="evaluation")
parser.add_argument("--lr_steps",default="40,50,60,70,80,90,95",type=str,
                    help="Steps for step_lr policy")

## network specific
parser.add_argument("--net_config",default="./models/configs/default_config.json",
                    type=str,
                    help="json file defining the network configuration")
parser.add_argument("--save_folder",default="exps/taskidAndName",
                    help="Directory for saving checkpoint models")
args = parser.parse_args()

def crop_data(img,lab):
    return img,lab

def ddp_print(content):
    if args.is_master:
        print(content)

def save_image_in_training(imgfiles,img,lab,logits,epoch,phase,idx):
    imgfile = imgfiles[idx]
    prefix = path_util.get_file_prefix(imgfile)
    with torch.no_grad():
        img_v = (unnormalize_normal(img[idx].numpy())[0]).astype(np.uint8)
        lab_v = (unnormalize_normal(lab[[idx]].numpy().astype(np.float))[0]).astype(np.uint8)

        logits = F.softmax(logits,dim=1).to(torch.device("cpu"))
        log_v = (unnormalize_normal(logits[idx,[1]].numpy())[0]).astype(np.uint8)

        if phase == "train":
            out_img_file = f"debug_epoch{epoch}_{prefix}_{phase}_img.tiff"
            out_lab_file = f"debug_epoch{epoch}_{prefix}_{phase}_lab.tiff"
            out_pred_file = f"debug_epoch{epoch}_{prefix}_{phase}_pred.tiff"
        else:
            out_img_file = f"debug_{epoch}_{prefix}_{phase}_img.tiff"
            out_lab_file = f"debug_{epoch}_{prefix}_{phase}_lab.tiff"
            out_pred_file = f"debug_{epoch}_{prefix}_{phase}_pred.tiff"

        sitk.WriteImage(sitk.GetImageFromArray(img_v),os.path.join(args.save_folder,out_img_file))
        sitk.WriteImage(sitk.GetImageFromArray(lab_v),os.path.join(args.save_folder,out_lab_file))
        sitk.WriteImage(sitk.GetImageFromArray(log_v),os.path.join(args.save_folder,out_pred_file))

def get_forward(img_d,lab_d,crit_c,crit_dice,model):
    logits = model(img_d) 
    if isinstance(logits,list):
        weights = [1./2**i for i in range(len(logits))]
        sum_weights = sum(weights)
        weights = [w / sum_weights for w in weights]
    else:
        weights = [1.]
        logits = [logits]

    loss_ce_items,loss_dice_items = [], []
    
    if args.use_robust_loss:
        


def load_dataset(phase,imgshape):
    dset = GenericDataset(args.data_file,phase=phase,imgshape=imgshape)
    ddp_print(f"Number of {phase} samples:{len(dset)}")
    ## distributedSampler
    if phase == "train":
        sampler = DistributedSampler(dset,shuffle=True)
    else:
        sampler = DistributedSampler(dset,shuffle=False)

    loader = tudata.DataLoader(dset,args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False,pin_memory=True,
                                sampler=sampler,
                                drop_last=True,
                                worker_init_fn=util.worker_init_fn)
    dset_iter = iter(loader)
    return loader,dset_iter

def evaluate(model,optimizer,crit_ce,crit_dice,imgshape):
    phase="test"
    val_loader,val_iter = load_dataset(phase,imgshape)
    loss_ce,loss_dice,loss = validate(model,val_loader,crit_ce,crit_dice,
                                        epoch=0,debug=True,num_image_save=10,phase=phase)
    ddp_print(f"Average loss_ce and loss_dice: {loss_ce:.5f} {loss_dice:.5f}")
        

def train(model,optimizer,crit_ce,crit_dice,imgshape):
    ## dataset preparing
    train_loader,train_iter = load_dataset("train",imgshape)
    val_loader,val_iter = load_dataset("val",imgshape)
    
    ## training process
    model.train()
    
    t0 = time.time()
    grad_scaler = GradScaler()
    debug = True
    debug_idx = 0
    best_loss_dice = 1.0e10
    for epoch in range(args.max_epochs):
        avg_loss_ce = 0
        avg_loss_dice = 0
        for it in range(args.step_per_epoch):
            try:
                img,lab,imgfiles,swcfiles = next(train_iter)
            except StopIteration:
                # let all process sync up before starting with a new epoch of training
                distrib.barrier()
                # reset the random seed,to avoid np.random & dataloader problem
                np.random.seed(args.seed + epoch)
                train_iter = iter(train_loader)
                img,lab,imgfiles,swcfiles = next(train_iter)
            
            # center croping for debug ,64x128x128 但是最后这里没有做centercrop
            img,lab = crop_data(img,data)
            img_d = img.to(args.device)
            lab_d = img.to(args.device)

            optimizer.zero_grad()
            if args.amp:
                with autocast():
                    loss_ces,loss_dices,loss,logits = get_forward(img_d,lab_d,crit_ce,crit_dice,model)
                    del img_d
                grad_scaler.scale(loss).backward()
                grad_sclaer.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm(model.parameters(),12)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss_ces,loss_dices,loss,logits = get_forward(img_d,lab_d,crit_ce,crit_dice,model)
                del img_d
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(),12)
                optimizer.step()
            
            avg_loss_ce += loss_ce[0]
            avg_loss_dice += loss_dices[0]

            # train statistics for debug afterward
            if it % args.print_frequency == 0:
                ddp_train(f"[{epoch}/{it}] loss_ce={loss_ces[0]}")












def main():
    ## keep track of master,usefule for IO
    args.is_master = args.local_rank == 0
    ## set device
    if args.cpu:
        args.device = util.init_device("cpu")
    else:
        args.device = util.init_device("args.local_rank")
    ## initialize group
    distrib.init_process_group(backend="nccl",init_method="env://")
    torch.cuda.set_device(args.local_rank)
    
    if args.deterministic:
        util.set_deterministic(deterministic=True,seed=args.seed)

    ## for output folder
    if args.is_master and not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    
    ## Network
    with open(args.net_config) as fp:
        net_configs = json.load(fp)
        print("Network configs:",net_configs)
        model = unet.UNet(**net_configs)
        ddp_print("\n" + "="*10 + "Network Structure" + "="*10)
        ddp_print(model)
        ddp_print("=" * 30 + "\n")

    # get the network downsizing informations
    ds_ratios = np.array([1,1,1])
    for stride in net_configs["stride_list"]:
        ds_ratios *= np.array(stride)
    args.ds_ratios = tuple(ds_ratios.tolist())
    
    model = model.to(args.device)
    if args.checkpoint:
        ## load checkpoint
        ddp_print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint,map_location={"cuda:0":f"cuda:{args.local_rank}"})
        model.load_state_dict(checkpoint.module.state_dict())
        del checkpoint

    ## convert to distributed data parallel model
    model = DDP(model,device_ids=[args.local_rank],
                output_device=args.local_rank)

    # optimizer & loss
    if args.checkpoint:
        args.lr /= 5
        # note:SGD is thought always better than Adam if training time 
        # is long enough
        optimizer = torch.optim.Adam(model.parameters(),args.lr,weight_decay=args.weight_decay,amsgrad=True) # whether to use the AMSGrad variant of this
            #algorithm from the paper `On the Convergence of Adam and Beyond`_
            #(default: False)
    else:
        optimizer = torch.optim.SGD(model.parameters(),args.lr,weight_decay=args.weight_decay,momentum=args.momentum,nesteroy=True) #使用Nesterov栋梁，默认False
    crit_ce = nn.CrossEntropyLoss(reduction="none").to(args.device)
    crit_dice = BinaryDiceLoss(smooth=1e-5,input_logits=False).to(args.device)
    
    args.imgshape = tuple(map(int,args.image_shape.split(",")))
    args.lr_steps = tuple(map(int,args.lr_steps.split(",")))

    # Print out the argument information
    ddp_print("Argument are:")
    ddp_print(f"{args}")

    if args.evaluation:
        evaluate(model,optimizer,crit_ce,crit_dice,args.imgshape)
    else:
        train(model,optimizer,crit_ce,crit_dice,args.imgshape)

if __name__ == "__main__":
    main()
















































