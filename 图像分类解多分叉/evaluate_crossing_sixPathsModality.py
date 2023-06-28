import os
import sys
import argparse
import numpy as np
import time
import json
import SimpleITK as sitk
import shutil
import pickle
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.data as tudata
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from neuronet.models import encoder
from neuronet.utils import util
from neuronet.utils.image_util import unnormalize_normal
from neuronet.datasets.generic_dataset import GenericDataset
from neuronet.loss.dice_loss import BinaryDiceLoss

import path_util

"""
parser = argparse.ArgumentParser(
    description='Solving Crossing problems') 
# data specific
parser.add_argument('--data_file', default='./data/XXX.pkl',
                    type=str, help='dataset split file')
# training specific
parser.add_argument('--batch_size', default=2, type=int,
                    help='')

def main():
    # keep track of master, useful for IO
    args.is_master = args.local_rank == 0
"""

def make_dataset(root, phase):
    #import ipdb; ipdb.set_trace()
    imgs = []
    train_inp_dir = 'train_input/data_splits.pkl'
    with open(os.path.join(root, train_inp_dir), 'rb') as fp:
        data_dict = pickle.load(fp)
    imgs.extend(data_dict[phase])
    """
    label_dir = 'label'
    n = len(os.listdir(os.path.join(root, train_inp_dir))) 
    inp = os.listdir(os.path.join(root, train_inp_dir))
    labels = np.loadtxt(os.path.join(root, label_dir)) 
    for i in range(n):
        img = inp[i]
        label = labels[i]
        imgs.append((img, label))
    """
    return imgs

class CrossingDataset(tudata.Dataset):
    def __init__(self,root,phase):
        imgs = make_dataset(root, phase)
        self.imgs = imgs

    def __getitem__(self, index):
        x_path, y = self.imgs[index]
        img_x = sitk.GetArrayFromImage(sitk.ReadImage((x_path)))
        return img_x, y, x_path
    def __len__(self):
        return len(self.imgs)

def validate(model, loss, data_iter, phase, debug=True, num_image_save=10):
    model.eval()
    num_saved = 0
    if num_image_save == -1:
        num_image_save = 9999
    
    loss_sum, n = 0.0, 0
    acc_sum_strict, acc_sum_loose, n_loose,n_strict = 0.0, 0.0, 0,0
    val_all = []
    #import ipdb; ipdb.set_trace()
    for X,y,_ in data_iter:
        X = X.float()
        y = y.float()
        inputs = X.to(device)
        labels = y.to(device)
        logits = net(inputs)
        m = nn.Sigmoid()
        #import ipdb; ipdb.set_trace()
        outputs = m(logits)
        l = loss(outputs, labels)
        loss_sum += l.item()
        n += 1

        maxIndex = outputs.argmax(dim=1)
        outputs[:] = 0
        outputs[torch.arange(outputs.shape[0]),maxIndex] = 1

        #output_all.append(outputs.detach().cpu().numpy())
        #y_all.append(y.detach().cpu().numpy())

        acc_sum_loose +=  (outputs == y.to(device)).float().sum().cpu().item()
        for i in range(len(y)):
            if (y[i].to(device) == outputs[i]).all():
                acc_sum_strict += 1
            val_all.append(outputs[i].detach().cpu().numpy())
            val_all.append(y[i].detach().cpu().numpy())
        n_loose += y.shape[0] * y.shape[1]
        n_strict += y.shape[0]
        
    return loss_sum / n, acc_sum_loose / n_loose, acc_sum_strict / n_strict, val_all
        


def evaluate_accuracy_train(outputs, y, net, epoch, device=None):
    # 对于train的时候算准确率的时候， 传进来的outputs是已经经过Sigmoid的logits，也就是概率了，
    #import ipdb; ipdb.set_trace()
    if device is None and isinstance(net, nn.Module):
        device = list(net.parameters)[0]
    acc_sum_strict, acc_sum_loose, n_loose,n_strict = 0.0, 0.0, 0, 0
    if isinstance(net, nn.Module):
        #net.eval()
        #logits = net(X.float().to(device))
        #m = nn.Sigmoid()
        #outputs = m(logits)
        #maxIndex = outputs.argmax(dim=1)
        #outputs[:,maxIndex] = -1
        #secIndex = outputs.argmax(dim=1)
        #outputs[:] = 0
        #outputs[:,torch.cat((maxIndex,secIndex))] = 1
        """
        if epoch == 10:
            import ipdb; ipdb.set_trace()
        """
        maxIndex = outputs.argmax(dim=1)
        outputs[:] = 0
        outputs[torch.arange(outputs.shape[0]),maxIndex] = 1

        acc_sum_loose +=  (outputs == y.to(device)).float().sum().cpu().item()
        for i in range(len(y)):
            if (y[i].to(device) == outputs[i]).all():
                acc_sum_strict += 1
    n_loose += y.shape[0] * y.shape[1]
    n_strict += y.shape[0]
        
    return acc_sum_loose / n_loose, acc_sum_strict / n_strict
    


def evaluate_accuracy(data_iter, net, epoch, device=None):
    #import ipdb; ipdb.set_trace()
    if device is None and isinstance(net, nn.Module):
        device = list(net.parameters)[0]
    acc_sum_strict, acc_sum_loose, n_loose,n_strict = 0.0, 0.0, 0,0
    #output_all = []
    #y_all = []
    val_all = []
    for X,y,_ in data_iter:
        if isinstance(net, nn.Module):
            net.eval()
            logits = net(X.float().to(device))
            m = nn.Sigmoid()
            outputs = m(logits)
            #import ipdb; ipdb.set_trace()
            maxIndex = outputs.argmax(dim=1)
            outputs[:] = 0
            outputs[torch.arange(outputs.shape[0]),maxIndex] = 1

            #output_all.append(outputs.detach().cpu().numpy())
            #y_all.append(y.detach().cpu().numpy())

            acc_sum_loose +=  (outputs == y.to(device)).float().sum().cpu().item()
            for i in range(len(y)):
                if (y[i].to(device) == outputs[i]).all():
                    acc_sum_strict += 1
                val_all.append(outputs[i].detach().cpu().numpy())
                val_all.append(y[i].detach().cpu().numpy())
            net.train()
        n_loose += y.shape[0] * y.shape[1]
        n_strict += y.shape[0]
        
    return acc_sum_loose / n_loose, acc_sum_strict / n_strict, val_all
    
    
def evaluate_loss(data_iter, net, loss, device=None):
    with torch.no_grad():
        if device is None and isinstance(net, nn.Module):
            device = list(net.parameters)[0] 
        loss_sum, n = 0.0, 0
        #import ipdb; ipdb.set_trace()
        for X,y,_ in data_iter:
            if isinstance(net, nn.Module):
                net.eval()
                X = X.float()
                y = y.float()
                inputs = X.to(device)
                labels = y.to(device)
                logits = net(inputs)
                m = nn.Sigmoid()
                outputs = m(logits)
                l = loss(outputs, labels)
                loss_sum += l.item()
                n += 1
            net.train()
    return loss_sum / n

def train_model(model, loss, optimizer, dataloaders, valloaders, output_file, num_epochs=20):
    #import ipdb; ipdb.set_trace()
    #output_all_val = []
    #label_all_val = []
    val = []
    val_all = []
    best_bceloss = 1.0
    for epoch in range(num_epochs):
        print('Epoch {:d}/{:d}'.format(epoch, num_epochs-1))
        print('-'*10)
        dt_size = len(dataloaders.dataset)
        epoch_loss = 0
        step = 0
        #import ipdb; ipdb.set_trace()
        for x, y, file_x in dataloaders:
            #x = x[:,:,:,::-1,:]
            if epoch % 10 == 0:
                save_index = np.random.randint(len(x),size=1)
                save_train_x = x[save_index]
                temp = np.zeros((32,32,32),dtype=np.uint8)
                for ii in range(len(save_train_x)):
                    file_prefix = os.path.splitext(os.path.split(file_x[save_index[ii]])[-1])[0]
                    prefix_train = f"train_epoch{epoch}_" + file_prefix
                    outfile =  os.path.join(output_file,prefix_train)
                    true_index = []
                    false_index = []
                    for i,obj in enumerate(file_prefix.split("_")[0]):
                        ## 下面的i要加1 因为save_train_x是7维的，第一维是原图
                        if obj == "1":
                            true_index.append(i+1)
                        else:
                            false_index.append(i+1)
                    #import ipdb; ipdb.set_trace()
                    save_train_img = np.concatenate((np.concatenate((save_train_x[ii][true_index[0]],save_train_x[ii][0],save_train_x[ii][true_index[1]]),axis=2),np.concatenate((save_train_x[ii][false_index[0]],temp,save_train_x[ii][false_index[1]]),axis=2),np.concatenate((save_train_x[ii][false_index[2]],temp,save_train_x[ii][false_index[3]]),axis=2)),axis=1)
                    save_train_img = save_train_img[:,::-1,:]
                    save_train_img_mip = np.max(save_train_img,axis=0)
                    sitk.WriteImage(sitk.GetImageFromArray(save_train_img),outfile + ".tiff")
                    sitk.WriteImage(sitk.GetImageFromArray(save_train_img_mip), outfile + ".jpg")
                
    
            #import ipdb; ipdb.set_trace()
            step += 1
            x = x.float()
            y = y.float()
            #import ipdb; ipdb.set_trace()
            inputs = x.to(device)
            labels = y.to(device)
            #print("labels:",labels)
            optimizer.zero_grad()
            logits = model(inputs)
            m = nn.Sigmoid()
            outputs = m(logits)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            epoch_loss += l.item()
            if step % 3 == 0:
                print('%d/%d, train_loss:%0.3f' % (step, (dt_size-1)//dataloaders.batch_size+1, l.item()))
                train_acc_loose,train_acc_strict = evaluate_accuracy_train(outputs, labels, model, epoch, device)
                print('epoch %d accuracy_loose_train:%0.3f accuracy_strict_train:%0.3f' % (epoch, train_acc_loose, train_acc_strict))

        # save model
        torch.save(model,os.path.join(output_file,"final_model.pt"))

        avg_val_loss = evaluate_loss(valloaders, model, loss, device)
        if avg_val_loss < best_bceloss:
            torch.save(model,os.path.join(output_file,"best_model.pt"))

        val_acc_loose,val_acc_strict, val = evaluate_accuracy(valloaders, model, epoch, device)

        #output_all_val.extend(output_val)
        #label_all_val.extend(label_val)
        #val_all.extend(output_all_val)
        #val_all.extend(label_all_val)
        #val_all.extend(val)
        print(f"[Val{epoch}] average BCELoss is {avg_val_loss:.3f}")
        print('epoch %d accuracy_loose_val:%0.3f accuracy_strict_val:%0.3f' % (epoch, val_acc_loose, val_acc_strict))
        #print('epoch %d loss:%0.3f' % (epoch, epoch_loss))
    #np.savetxt("./exps/exp002/val.txt", val_all, fmt="%d")
    return model

if __name__ == "__main__":
    #import ipdb; ipdb.set_trace()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_config = "/home/yjz/Project/Auto-tracing/neuronet/neuronet_new_0519_crossing_threeUniteModality/neuronet/models/configs/default_config.json"
    output_file = "/home/yjz/Project/Auto-tracing/neuronet/neuronet_new_0519_crossing_threeUniteModality/neuronet/exps/exp003"
    model_file = "/home/yjz/Project/Auto-tracing/neuronet/neuronet_new_0519_crossing_threeUniteModality/neuronet/exps/exp003/best_model.pt"
    batch_size = 64
    crossing_dataset = CrossingDataset("/home/yjz/Project/Auto-tracing/neuronet/neuronet_new_0519_crossing_threeUniteModality/neuronet/data/task001_onlyForTest", "test")
    dataloaders = tudata.DataLoader(crossing_dataset,batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    #val_dataset = CrossingDataset("/home/yjz/Project/Auto-tracing/neuronet/neuronet_new_0519_crossing_sixPathsModality/neuronet/data/task008_optimizeExtractCrossingRegion" , "val")
    #valloaders = tudata.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    with open(net_config) as fp:
        net_configs = json.load(fp)
        print("Network configs: ", net_configs)
        net = encoder.UNet(**net_configs)
        net = net.to(device)
        print("\n" + "="*10 + "Network Structure" + "="*10)
        print(net)
        print("=" * 30 + "\n")
    #loss = torch.nn.CrossEntropyLoss()
    #import ipdb; ipdb.set_trace()
    checkpoint = torch.load(model_file, map_location=device)
    net.load_state_dict(checkpoint.state_dict())
    loss = nn.BCELoss()
    loss, acc_loose, acc_strict,_ = validate(net, loss, dataloaders, phase="test", debug=True, num_image_save=10)
    print(loss,acc_loose, acc_strict)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.99, weight_decay=5e-4) #正则化从3e-5调的更大一点
    #model = train_model(net, loss, optimizer, dataloaders, valloaders, output_file, num_epochs=50)
    
