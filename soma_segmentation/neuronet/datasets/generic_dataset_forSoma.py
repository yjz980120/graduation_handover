import numpy as np
import pickle
import torch.utils.data as tudata
import SimpleITK as sitk
import torch
import sys

from swc_handler import parse_swc
from neuronet.augmentation.generic_augmentation_forSoma import InstanceAugmentation
from neuronet.datasets.swc_processing import trim_swc,swc_to_image,trim_out_of_box

# To avoid the recursionlimit error, maybe encountered in trim_swc
sys.setrecursionlimit(30000)

class GenericDataset(tudata.Dataset):
    def __init__(self,split_file,phase="train",imgshape=(256,512,512)):
        self.data_list = self.load_data_list(split_file,phase)
        self.imgshape = imgshape
        print(f"Image shape of {phase}: {imgshape}")

        ## augmentation
        self.augment = InstanceAugmentation(p=0.2,imgshape=imgshape,phase=phase)

    @staticmethod
    def load_data_list(split_file,phase):
        with open(split_file,"rb") as fp:
            data_dict = pickle.load(fp)
        return data_dict[phase]
    
    def __getitem__(self,index):
        img,gt,imgfile,labfile = self.pull_item(index)
        return img,gt,imgfile,labfile
    
    def __len__(self):
        return len(self.data_list)

    def pull_item(self,index):
        imgfile,labfile,spacing = self.data_list[index]
        # parse, image should in [c,z,y,x] format
        img = np.load(imgfile)["data"]
        if img.ndim == 3:
            img = img[None]
        lab = np.load(labfile)["data"]
        # random augmentation
        img,lab,_ = self.augment(img,lab,spacing)
        # convert swc to image
        # firstly trim_swc via deleting out-of-box points
        return torch.from_numpy(img.astype(np.float32)),torch.from_numpy(lab.astype(np.uint8)),imgfile,labfile


if __name__=="__main__":
    split_file = ""
    idx = 2
    imgshape = (256,512,512)
    dataset = GenericDataset(split_file,"train",imgshape)
    img,lab = dataset.pull_item(idx) 
    
