#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#
#   Filename     : preprocess.py
#   Author       : Yufeng Liu
#   Date         : 2021-05-15
#   Description  : This package tries to standardize the input image,
#                  for lowerize the burden when training, including:
#                  - resampling
#                  - normalization
#                  - format conversion
#                  - dataset splitting
#
#================================================================

import os,glob
import numpy as np
from skimage.io import imread,imsave
from skimage.transform import resize
from scipy.ndimage.interpolation import map_coordinates
from copy import deepcopy
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from multiprocessing.pool import Pool
import pickle

from swc_handler import parse_swc,write_swc
from path_util import get_file_prefix

from neuronet.utils.image_util import normalize_normal,unnormalize_normal
from neuronet.datasets.swc_processing import soma_labelling,trim_swc,load_spacing


def load_data(data_dir,spacing_file,is_train=True):
    # load the spacing file
    spacing_dict = load_spacing(spacing_file)
    data_list = []
    for imgfile in glob.glob(os.path.join(data_dir,"*")):
        brain_id = '18454'
        spacing = spacing_dict[int(brain_id)]
        #label_dir = os.path.join(data_dir,"soma_segmentation_training_segmentations_mirrorImage",brain_id+"_seg_watershed_SEU200")
        prefix = get_file_prefix(imgfile)
        if is_train:
            label_dir = os.path.join(data_dir,"lab",brain_id)
            label_file = os.path.join(label_dir,f"{prefix}.tiff")
        else:
            label_file = None
            
          
        data_list.append((imgfile,label_file,spacing))
    return data_list

def calc_spacing_anisotropy(spacing):
    """
    spacing in (z,y,x) format
    """
    assert spacing[1] == spacing[2] and spacing[1] <= spacing[0],"Spacing in X- and Y-dimension must be the same, and the must smaller than Z-axis"
    spacing_multi = 1.0 * spacing[0] / spacing[2]
    return spacing_multi
    
class GenericPreprocessor(object):
    def __init__(self,separate_z_thresh=2,label_soma=False,lab_hasSomaLab=False):
        self.separate_z_thresh = separate_z_thresh
        self.label_soma = label_soma
        self.lab_hasSomaLab = lab_hasSomaLab

    def remove_nans(self,data):
        # inplace modification of nans
        data[np.isnan(data)] = 0
        return data

    def resampling(self,data,lab=None,orig_spacing=None,target_spacing=None,order=3):
        assert data.ndim == 4, "image must in 4 dimension:c,z,y,x"
        # whether resize separately for Z-axis
        separate_z = False
        if calc_spacing_anisotropy(orig_spacing) > self.separate_z_thresh or calc_spacing_anisotropy(target_spacing) > self.separate_z_thresh:
            separate_z  =True
    
        #dtype = data.dtype ##source code,but I think this is redundant
        data = data.astype(np.float32) ## float32 is sufficient
        shape = np.array(data[0].shape)
        new_shape = np.round(((np.array(orig_spacing) / np.array(target_spacing)).astype(np.float32) * shape)).astype(int)
        #import ipdb; ipdb.set_trace()
        if np.all(shape == new_shape):
            print("no resampling necessary")
            return data,lab
        else:
            print("resampling:")
    
        if separate_z:
            z_axis = 0
            new_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[z_axis]):
                    reshaped_data.append(resize(data[c,slice_id],new_shape[1:],order=order,mode="edge"))
                reshaped_data = np.stack(reshaped_data,z_axis)
                if shape[z_axis] != new_shape[z_axis]:
                    # resizing in z dimension, code from sklearn's resize
                    rows,cols,dim = new_shape[0],new_shape[1],new_shape[2]
                    orig_rows,orig_cols,orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows,map_cols,map_dims = np.mgrid[:rows,:cols,:dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows,map_cols,map_dims])
                    new_data.append(map_coordinates(reshaped_data,coord_map,order=0,cval=0,mode="nearest")[None])
                else:
                    new_data.append(reshaped_data[None])
            new_data = np.vstack(new_data)
        else:
            print("no separate z,order",order)
            new_data = []
            for c in range(data.shape[0]):
                new_data.append(resize(data[c],new_shape,order,cval=0,mode="edge"))
            new_data = np.vstack(new_data)
        
        #import ipdb;ipdb.set_trace()
        if lab is not None:
            lab = lab.astype(np.float32) #如果这里自己不变的话 下面resize会变成np.float64，但是sitk.WriteImage()还不能保存超过float32的（也就是float64无法保存）
            if separate_z:
                new_lab = []
                reshaped_lab = []
                for slice_id in range(shape[z_axis]):
                    reshaped_lab.append(resize(lab[slice_id],new_shape[1:],order=0,mode="edge")) #order=0 means nearest interpolation
                reshaped_lab = np.stack(reshaped_lab,z_axis)
                if shape[z_axis] != new_shape[z_axis]:
                    # resizing in z dimension, code from sklearn's resize
                    rows,cols,dim = new_shape[0],new_shape[1],new_shape[2]
                    orig_rows,orig_clos,orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows,map_cols,map_dims = np.mgrid[:rows,:cols,:dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows,map_cols,map_dims])
                    new_lab = map_coordinates(reshaped_lab,coord_map,order=0,cval=0,mode="nearest")
                else:
                    new_lab = reshaped_lab
            #import ipdb;ipdb.set_trace()

            return new_data,new_lab
        else:
            return new_data,lab
    

    def _preprocess_sample(self,imgfile,labfile,imgfile_out,labfile_out,spacing,target_spacing):
        print(f"--> Processing for image:{imgfile}")
        ## load the image and annotated tree
        #import ipdb; ipdb.set_trace()
        image = sitk.GetArrayFromImage(sitk.ReadImage(imgfile))
        ## manually labelling the soma
        ## if use for soma seg,please don not do this 
        if self.label_soma:
            image = soma_labelling(image,z_ratio=0.3,r=9)
        
        if image.ndim == 3:
            image = image[None]
        lab = None
        if labfile is not None:
            lab = sitk.GetArrayFromImage(sitk.ReadImage(labfile))
            #import ipdb;ipdb.set_trace()
            #if not self.lab_hasSomaLab: soma seg 暂时不需要对soma进行label 后面可以看增加了空心aug之后对解决空心soma是否有帮助，如果解决的不好 试试在中间加soma lab  可能对空心有帮助
                #lab = soma_labelling(lab,z_ratio=0.3,r=9)
            if lab.max() > 1:
                lab[lab>1] = 1 #因为现在已经是preprocess了，也就是送进去训练前的最后一步，所以这里就要把lab转换成我们需要的标签了，不能再像原来一样有的值是0，有的是255，就应该是0和1两类标签，而且soma_lab那里又给soma加了方框标记，是220，所以也要转换成1分类
  
        # remove nans
        image = self.remove_nans(image)
        # resampling to target spacing
        #import ipdb;ipdb.set_trace()
        image,lab = self.resampling(image,lab,spacing,target_spacing)
        # normalize the image
        image = normalize_normal(image,mask=None)
        # write the image and tree as well
        if not os.path.exists(os.path.join(imgfile_out)):
            np.savez_compressed(imgfile_out,data=image.astype(np.float32))
        #write the img as tiff format for visual to check
        #import ipdb; ipdb.set_trace()
         
        if not os.path.exists(os.path.join(imgfile_out[:-4]+".tiff")):
            sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(unnormalize_normal(image)).astype(np.uint8)),imgfile_out[:-4]+".tiff")
        
        if lab is not None:
            # write the lab as tiff format for visual to check
           
            if not os.path.exists(os.path.join(labfile_out[:-4]+".tiff")):
                #import ipdb;ipdb.set_trace()
                sitk.WriteImage(sitk.GetImageFromArray(lab.astype(np.uint8) * 255),labfile_out[:-4]+".tiff")
            
            if not os.path.exists(os.path.join(labfile_out)):
                np.savez_compressed(labfile_out,data=lab.astype(np.uint8))

        else:
            if not os.path.exists(os.path.join(labfile_out[:-4]+".tiff")):
                #import ipdb;ipdb.set_trace()
                sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(unnormalize_normal(image)).astype(np.uint8)),labfile_out[:-4]+".tiff")
            
            if not os.path.exists(os.path.join(labfile_out)):
                np.savez_compressed(labfile_out,data=image.astype(np.float32))



    @staticmethod
    def get_target_spacing(spacings):
        # assume spacing in format[z,y,x]
        spacings = sorted(spacings,key=lambda x:x.prod())
        target_spacing = spacings[len(spacings) // 2]
        return target_spacing


    def run(self,data_dir,spacing_file,output_dir,is_train=True,num_threads=8):
        print("Processing for dataset, should be run at least once for each dataset!")
        # get all files
        
        #import ipdb; ipdb.set_trace()
        data_list = load_data(data_dir,spacing_file,is_train=is_train)
        print(f"Total number of samples found:{len(data_list)}")
        # estimate the target spacing
        spacings = [spacing for _,_,spacing in data_list]
        #self.target_spacing = self.get_target_spacing(spacings)
        self.target_spacing = (1.0,0.23,0.23)

        maybe_mkdir_p(output_dir)
        ## execute preprocessing
        args_list = []
        for imgfile,labfile,spacing in data_list:
            prefix = get_file_prefix(imgfile)
            #print(imgfile,swcfile)
            imgfile_out = os.path.join(output_dir,f"{prefix}_img.npz")
            labfile_out = os.path.join(output_dir,f"{prefix}_lab.npz")
            args = imgfile,labfile,imgfile_out,labfile_out,spacing,self.target_spacing
            args_list.append(args)

        #self._preprocess_sample(*args_list[-2])
        #execute in parallel
        pt = Pool(num_threads)
        pt.starmap(self._preprocess_sample,args_list)
        pt.close()
        pt.join()

    def dataset_split(self,task_dir,val_ratio=0.1,test_ratio=0.1,seed=1024,img_ext="npz",lab_ext="npz"):
        samples = []
        for imgfile in glob.glob(os.path.join(task_dir,f"*img.{img_ext}")):
            labfile = f"{imgfile[:-(len(img_ext)+4)]}lab.{lab_ext}" ## 4 is img.
            samples.append((imgfile,labfile,self.target_spacing))
        num_tot = len(samples)
        num_val = int(round(num_tot * val_ratio))
        num_test = int(round(num_tot * test_ratio))
        num_train = num_tot - num_val - num_test
        print(f"Number of samples of total/train/val/test are {num_tot}/{num_train}/{num_val}/{num_test}")

        np.random.seed(seed)
        np.random.shuffle(samples)
        splits = {}
        splits['train'] = samples[:num_train]
        splits['val'] = samples[num_train:num_train+num_val]
        splits['test'] = samples[-num_test:]
        # write to file
        with open(os.path.join(output_dir, 'data_splits.pkl'), 'wb') as fp:
            pickle.dump(splits, fp)

    def get_testdata(self,task_dir,val_ratio=0.1,test_ratio=0.1,seed=1024,img_ext="npz",lab_ext="npz"):
        samples = []
        for imgfile in glob.glob(os.path.join(task_dir,f"*img.{img_ext}")):
            labfile = f"{imgfile[:-(len(img_ext)+4)]}lab.{lab_ext}" ## 4 is img.
            samples.append((imgfile,labfile,self.target_spacing))
        
        splits = {}
        splits['test'] = samples[:]
        # write to file
        with open(os.path.join(output_dir, 'data_splits.pkl'), 'wb') as fp:
            pickle.dump(splits, fp)



if __name__ == "__main__":
    ## 注意这次用的complex soma 只有四个 都是17052里面的，所以为了和之前训练样本的target_spacing保持一致，就直接令target_spacing=之前训练的target_spacing。单独构造test的时候 应该就是要手动指定target_spacing
    data_dir = "/media/yjz/My Book/Dr_wangyimin/mini_191797_crop_soma/tiff" ##注意这里用的complex soma 没有lab 所以img 和lab  都是同一个img，如果有lab的话要到上面改一下，指定lab的文件夹。或者由于是test阶段，可以直接指定is_train=False, False的时候就默认不读入lab 是None. ！！！ 还是不能指定is_train为False  因为后面evaluate的时候生成pred的同时 还要作evaluate 需要用到lab，后面可以自己改一个  改成单独用于infer的
    #data_dir = "/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/data/task006_withMultiSomaOfflineAug_removeSomaBadMultiSoma/complexSomaForInfer_oriData"
    spacing_file = "/home/yjz/Projects/Auto_tracing/neuronet_forSoma/scripts/AllbrainResolutionInfo.csv"

    #output_dir = "/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/exps/exp003/exp003_predictMostComplexSoma"
    output_dir = "/media/yjz/My Book/Dr_wangyimin/mini_191797_crop_soma/segmentation_preprocess"
    is_train = False
    num_threads = 8 #一般可以用8，但是有时候服务器很卡，就用4
    gp = GenericPreprocessor()
    #data_list = load_data(data_dir,spacing_file,is_train=is_train)
    # estimate the target spacing
    #spacings = [spacing for _,_,spacing in data_list]
    #gp.target_spacing = gp.get_target_spacing(spacings)
    gp.run(data_dir,spacing_file,output_dir,is_train=is_train,num_threads=num_threads)
    #gp.dataset_split(output_dir,val_ratio=0.06,test_ratio=0.03,seed=1024,img_ext="npz",lab_ext="npz")
    gp.get_testdata(output_dir,val_ratio=0.06,test_ratio=0.03,seed=1024,img_ext="npz",lab_ext="npz")




