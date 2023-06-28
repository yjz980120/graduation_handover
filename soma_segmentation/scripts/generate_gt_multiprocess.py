from swc_handler import parse_swc
from neuronet.datasets.swc_processing import swc_to_image,trim_out_of_box
from path_util import get_file_prefix
import SimpleITK as sitk
import os
import glob
from multiprocessing.pool import Pool


def generateGt(swc_path,gt_outfile,prefix,imgshape):
    #import ipdb;ipdb.set_trace()
    print('==> Processing for image: {:s}'.format(prefix))
    tree = parse_swc(swc_path)
    tree = trim_out_of_box(tree,imgshape,True)
    gt = swc_to_image(tree,imgshape=imgshape)
    #if gt.max()<50:## 为了造成不必要的麻烦，只在需要显示看效果的时候再*255 否则的话gt=255很麻烦，因为我们现在做二分类，就只有0 和 1 两种值 两个类别 255的话 会在后面训练的时候出问题，除非在训练的时候加上if gt.max()>1 gt[gt>1]=1  把255再给变回1（第1个类别，对应的0就是第0个类别），但是这样的话 在训练的时候徒增一个计算，本来现在的计算就已经很慢了，再加一个这个计算，得不偿失
    #gt *= 255
            
    sitk.WriteImage(sitk.GetImageFromArray(gt),gt_outfile) 

if __name__ == "__main__":
    imgshape = (256,512,512)
    swc_dir = '/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/dendriteImageSecR_mini/swc/'
    lab_dir = '/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/dendriteImageSecR_mini/labelIS_semantic_correct_sitkandLabels/'     
    args_list = []
    num_threads = 16
    for swc_in_path in glob.glob(os.path.join(swc_dir,"*")):
        brain_id = os.path.split(swc_in_path)[-1]
        print(f'##### Brain: {brain_id}')
        label_path = os.path.join(lab_dir + brain_id)
        print(swc_in_path)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        for swc_path in glob.glob(os.path.join(swc_in_path,"*.swc")):
            prefix = get_file_prefix(swc_path)
            gt_outfile = os.path.join(label_path,"{:s}.tiff".format(prefix))
            args_list.append[(swc_path,gt_outfile,prefix,imgshape)]
    
    pt = Pool(num_threads)
    pt.starmap(generateGt,args_list)
    pt.close()
    pt.join()
