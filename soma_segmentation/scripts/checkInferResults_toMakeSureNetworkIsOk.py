import glob
import os
import SimpleITK as sitk
from path_util import get_file_prefix
#from neuronet.datasets.swc_processing import soma_labelling
import numpy as np


def check(infer_dir):
    #import ipdb;ipdb.set_trace()
    num = 0
    for img_file in glob.glob(os.path.join(infer_dir,"*img_test_img.tiff")):
        num += 1
        if num > 10:
            break
        #img_file in os.listdir(img_path):
        prefix = get_file_prefix(img_file)
        lab_prefix = f"{prefix[:-3]}pred" ##其实最简单的就是现在这样Img也用predictTest里面的prefix
            #if lab_prefix not in os.list(os.path.join(infer_dir,"*img_test_pred.tiff")):
                  #continue
        print("check_file:",prefix)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
        #lab_prefix = f"debug_{prefix}_test_pred"
        lab_file = os.path.join(infer_dir,lab_prefix+".tiff")
        lab = sitk.GetArrayFromImage(sitk.ReadImage(lab_file))
        imgMask = img * (lab == 0)
        print("img!=0 num:",np.sum(imgMask!=0))
        print("img!=0 ratio:",np.sum(imgMask!=0) / np.prod(img.shape))
        print("img_mean:",img.mean())
        print("img_std:",img.std())
        print("imgMask_mean:",(imgMask).mean())
        print("imgMask_std:",imgMask.std())



if __name__ == "__main__":
    #img_dir = "/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/dendriteImageSecR_AllData/tiff"
    infer_dir = "/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet_reproduction/neuronet/exps/exp006/exp006_predictTest"
    check(infer_dir)



            
