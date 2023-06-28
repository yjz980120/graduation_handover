import torch
import torch.nn as nn
import torch.nn.functional as F
from .soft_skeletonization import soft_skel
#from soft_skeletonization import soft_skel
from neuronet.loss.dice_loss import BinaryDiceLoss
from neuronet.utils.image_util import normalize_normal

class Soft_cldice(nn.Module):
    def __init__(self,iters_=3,smooth=1.,input_logits=True):
        super(Soft_cldice,self).__init__()
        self.iters = iters_
        self.smooth = smooth
        self.input_logits = input_logits

    def forward(self,logits,gt_float):
        assert logits.shape[0] == gt_float.shape[0], "batch size error!"
        
        #print("logits.shape!!!:",logits.shape)
        #print("logits.max!!!:",logits.max())
        if self.input_logits:
            probs = F.softmax(logits, dim=1)[:,1]    # foreground
        else:
            probs = logits[:,1]
        
        #import ipdb;ipdb.set_trace()
        #probs = logits
        #print("probs.shape!!!:",probs.shape)
        #print("probs.max!!!:",probs.max())
        skel_probs = soft_skel(probs,self.iters)
        with torch.no_grad():
            skel_gt_float = soft_skel(gt_float,self.iters)
        if skel_probs.ndim == 5:
            skel_probs = skel_probs.squeeze(0)
        if skel_gt_float.ndim == 5:
            skel_gt_float = skel_gt_float.squeeze(0)
        if skel_probs.ndim == 3: ##很奇怪，不知道为什么gt_float过了soft_skel函数之后变成3维了，但是明明进去之前维度和类型都与probs一致
            skel_probs = skel_probs.unsqueeze(0)
        if skel_gt_float.ndim == 3:
            skel_gt_float = skel_gt_float.unsqueeze(0)
        

        #print("before_probs.shape!!!!!:",probs.shape)
        #print("before_gt_float.shape!!!!!:",gt_float.shape)
        #print("before_probs_dtype!!!:",probs.dtype)
        #print("before_gt_float_dtype!!!:",gt_float.dtype)
        #print("before_skel_probs.shape!!!!!:",skel_probs.shape)
        #print("before_skel_gt_float.shape!!!!!:",skel_gt_float.shape)
        probs = probs.contiguous().view(probs.shape[0], -1)
        gt_float = gt_float.contiguous().view(gt_float.shape[0], -1)
        #skel_probs = skel_probs.contiguous().view(skel_probs.shape[0],-1).to(probs.device)
        #skel_gt_float = skel_gt_float.contiguous().view(skel_gt_float.shape[0],-1).to(gt_float.device)
        
        skel_probs = skel_probs.contiguous().view(skel_probs.shape[0],-1)
        skel_gt_float = skel_gt_float.contiguous().view(skel_gt_float.shape[0],-1)


        #print("skel_probs_device!!!!!!:",skel_probs.device)
        #print("probs_device!!!!!::",probs.device)
        #print("after_probs.shape!!!!!:",probs.shape)
        #print("after_gt_float.shape!!!!!:",gt_float.shape)
        #print("after_skel_probs.shape!!!!!:",skel_probs.shape)
        #print("after_skel_gt_float.shape!!!!!:",skel_gt_float.shape)

        tprec = (torch.sum(torch.multiply(skel_probs,gt_float),dim=1)+self.smooth) / (torch.sum(skel_probs,dim=1)+self.smooth)
        tsens = (torch.sum(torch.multiply(skel_gt_float,probs),dim=1)+self.smooth) / (torch.sum(skel_gt_float,dim=1)+self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        alpha = 0.4 ## cldice在训练中占的比例
        return cl_dice * alpha

class Soft_dice_cldice(nn.Module):
    def __init__(self,iter_=3,alpha=0.5,smooth=1.,input_logits=True):
        super(Soft_dice_cldice,self).__init__()
        self.iters = iters_
        self.smooth = smooth
        self.alpha = alpha
        self.input_logits = input_logits
       
    def forward(self,logits,gt_float):
        assert logits.shape[0] == gt_float.shape[0], "batch size error!"
        if self.input_logits:
            probs = F.softmax(logits, dim=1)[:,1]    # foreground
        else:
            probs = logits[:,1]

        probs = probs.contiguous().view(probs.shape[0], -1)
        gt_float = gt_float.contiguous().view(gt_float.shape[0], -1)
        crit_dice = BinaryDiceLoss(smooth=1e-5, input_logits=False)
        loss_dice = crit_dice(probs,gt_float)
        crit_cldice = Soft_cldice(iters_=3,smooth=1e-5,input_logits=False)
        loss_cldice = crit_cldice(probs,gt_float)
        return (1.0-self.alpha)*loss_dice + self.alpha*loss_cldice

if __name__ == "__main__":
    import SimpleITK as sitk
    import os
    import numpy as np
    path = "/home/yjz/Projects/Auto_tracing/neuronet/exps/exp002/exp002_predict"
    lab_file = "debug_9996_5435_2511_test_lab.tiff"
    pred_file = "debug_9996_5435_2511_test_pred.tiff"
    lab = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path,lab_file)))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path,pred_file)))
    
    import ipdb;ipdb.set_trace()
    lab = lab[None]
    pred = pred[None]
    lab = lab.astype(np.float32)
    pred = pred.astype(np.float32)
    lab /= 255
    pred /= 255
    #lab = normalize_normal(lab)
    #pred = normalize_normal(pred)
    lab = torch.from_numpy(lab)
    pred = torch.from_numpy(pred)
    lab = lab.unsqueeze(0)
    pred = pred.unsqueeze(0)
    crit_cldice = Soft_cldice(smooth=1e-5,input_logits=False)
    crit_cldice(pred,lab)
    print("cldice!!!:",crit_cldice(pred,lab))


























