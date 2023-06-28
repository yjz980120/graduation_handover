import numpy as np
import SimpleITK as sitk
import os
from glob import glob
import time
import copy 
import multiprocessing
import random
from neuronet.utils import image_util

"""
def move_image_toCentral(img,lab):
    !!!!这个函数不应该放在增强这里，就放在外面应该先在外面把图移动过了，再送进来增强什么的
    # caculate center of gravity, and then move the image to central
    # if you use neuronet for soma segmentation,you may need to use this augmantation
    # if you use neuronet for neuron segmentation,you may do not need to use this, because neuron image is much bigger than soma image,inaccurate marker impact little, can be ingored. But soma image is small ,we need to move soma to the central of the image as much as we can
    temp = np.ones(img.shape)
    # 下面的xyz是为了分别取出index对应的x y z 找到的规律就是下面这样，可以后面看看有没有更好的方法
    y = temp * np.arange(lab.shape[2])
    x = (temp.T * np.arange(lab.shape[0]).T).T
    temp_z = np.ones(lab.shape[1:])
    temp_2d_z = list((temp_x.T * np.arange(lab.shape[1]).T).T)
    z = np.stack(temp_2d_x * lab.shape[0],axis=0).reshape(lab.shape[0],lab.shape[1],lab.shape[2])
    xc = int(round(np.sum(lab * x) / np.sum(lab)))
    yc = int(round(np.sum(lab * y) / np.sum(lab)))
    zc = int(round(np.sum(lab * z) / np.sum(lab)))
    x_ori = int(round(lab.shape[2] / 2))
    y_ori = int(round(lab.shape[1] / 2))
    z_ori = int(round(lab.shape[0] / 2))
    x_d = x_ori - abs(x_ori - xc)
    y_d = y_ori - abs(y_ori - yc)
    z_d = z_ori - abs(z_ori - zc)
    img = img[:,zc-z_d:zc+z_d,yc-y_d:yc+y_d,xc-x_d:xc+xd]
    lab = lab[:,zc-z_d:zc+z_d,yc-y_d:yc+y_d,xc-x_d:xc+xd]
    img = np.pad(img,((0,0),((z_ori-z_d)/2,(z_ori-z_d)/2),((z_ori-z_d)/2,(y_ori-y_d)/2),((x_ori-x_d)/2,(x_ori-x_d)/2)),mode="edge")
    lab = np.pad(lab,(((z_ori-z_d)/2,(z_ori-z_d)/2),((z_ori-z_d)/2,(y_ori-y_d)/2),((x_ori-x_d)/2,(x_ori-x_d)/2)),mode="edge")
    sitk.WriteImage(sitk.GetImageFromArray(img),)
    
    return img,lab
"""    
"""
class brightAndDark(input_dir,mask_dir,outputBD_dir,ratio,bright_reduceRank):
    for f in glob(os.path.join(input_dir,"*.tiff")):
        filename = os.path.split(f)[-1]
        prefix = os.path.splitext(filename)[0]
        if os.path.exists(os.path.join(outputBD_dir,prefix+"_BD.tiff")):
            continue
        input_data = sitk.GetArrayFromImage(sitk.ReadImage(f))
        mask_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir,filename)))
        #hollow_data = copy.deepcopy(mask_data)
        output_data = copy.deepcopy(input_data)
        #hollow_value = np.min(hollow_data)
        mask_candidateMatrix = [True]*int(10*ratio) + [False]*int(10-10*ratio)
        np.random.shuffle(mask_candidateMatrix)
        all_maskMatrix = np.random.choice(mask_candidateMatrix,size=input_data.shape)
        maskMatrix = all_maskMatrix & (mask_data > 0)
        output_data[maskMatrix==True] = output_data[maskMatrix==True] / bright_reduceRank
        sitk.WriteImage(sitk.GetImageFromArray(output_data),os.path.join(outputBD_dir,prefix+"_BD.tiff"))
"""
    
"""
def randomHollow(input_dir,mask_dir,outputHOL_dir,ratio):
    for f in glob(os.path.join(input_dir,"*.tiff")):
        filename = os.path.split(f)[-1]
        prefix = os.path.splitext(filename)[0]
        if os.path.exists(os.path.join(outputBD_dir,prefix+"_HOL.tiff")):
            continue
        input_data = sitk.GetArrayFromImage(sitk.ReadImage(f))
        mask_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir,filename)))
        hollow_data = copy.deepcopy(mask_data)
        output_data = copy.deepcopy(input_data)
        hollow_value = np.min(hollow_data)
        mask_candidateMatrix = [True]*int(10*ratio) + [False]*int(10-10*ratio)
        np.random.shuffle(mask_candidateMatrix)
        all_maskMatrix = np.random.choice(mask_candidateMatrix,size=input_data.shape)
        maskMatrix = all_maskMatrix & (mask_data > 0)
        output_data[maskMatrix==True] = hollow_value
        sitk.WriteImage(sitk.GetImageFromArray(output_data),os.path.join(outputHOL_dir,prefix+"_HOL.tiff"))
"""

"""
def centerHollow(input_dir,mask_dir,outputHOL_dir,ratio):
    for f in glob(os.path.join(input_dir,"*.tiff")):
        filename = os.path.split(f)[-1]
        prefix = os.path.splitext(filename)[0]
        if os.path.exists(os.path.join(outputBD_dir,prefix+"_CTHOL.tiff")):
            continue
        input_data = sitk.GetArrayFromImage(sitk.ReadImage(f))
        mask_data = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_dir,filename)))
        hollow_data = copy.deepcopy(mask_data)
        output_data = copy.deepcopy(input_data)
        hollow_value = np.min(hollow_data)
        mask_candidateMatrix = [True]*int(10*ratio) + [False]*int(10-10*ratio)
        np.random.shuffle(mask_candidateMatrix)
        all_maskMatrix = np.random.choice(mask_candidateMatrix,size=input_data.shape)
        maskMatrix = all_maskMatrix & (mask_data > 0)
        output_data[maskMatrix==True] = hollow_value
        sitk.WriteImage(sitk.GetImageFromArray(output_data),os.path.join(outputHOL_dir,prefix+"_CTHOL.tiff"))
"""


#def random_circle():


class AbstractTransform(object):
    def __init__(self,p=0.5):
        self.p = p


class RandomCircleHollow(AbstractTransform):
    def __init__(self,p=0.7):
        super(RandomCircleHollow,self).__init__(p)
        #self.

    def __call__(self,img,lab,spacing):
        if np.random.random() < self.p:
            mask = np.argwhere(lab>0)
            bg = np.argwhere(lab==0)
            num = mask.shape[0] # shape[1]是3 对应了lab中大于0的元素的xyz坐标
            status = True
            #import ipdb;ipdb.set_trace()
            while status:
                #mask = np.argwhere(lab>0)
                #num = mask.shape[0] # shape是3 对应了lab中大于0的元素的xyz坐标
                centercir = mask[random.sample(range(num),k=1)] #从lab的mask中随机选一个点作为圆心
                #bg = np.argwhere(lab==0)
                radius_all = np.sum((centercir-bg)**2,axis=1)#如果直接检测lab的mask边缘的话会很麻烦，然后我们的半径肯定是小于选取的centercir(圆心)到边缘最小的点，但是边缘检测比较麻烦，且费时间，所以转换成求圆心与边缘外面那一圈背景的距离，与边缘紧挨着的那一圈背景的距离也就是，也就是圆心到背景最小的那几个距离。 而我们需要的是最小，所以这里也取最小，但是因为是边缘外侧的一圈，所以一定比圆心直接到边缘要大一点，但是最多就是大斜上方一个pixel的距离，也就是根号2，但是一般肯定不到根号2，也就是不会在斜上方，应该是在与边缘紧紧水平相邻的一个pixel，所以增大的距离肯定效于根号2，但是也是个大于1的根号，比较麻烦算，这里简单一点就用根号2替代
                radius = sorted(radius_all)[0] - 2 if sorted(radius_all)[0] - 2>0 else sorted(radius_all)[0] - 1 #但是不可能有根号2个pixel，所以我们就减去2，如果减去2小于0，那就减去1 ，如果减去1还小于零就证明取到了边缘上或者距离边缘只有1个pixel（这些都是不合适的），那就重新取
                if radius > 0:
                    status = False
            distant = np.sum((centercir-mask)**2,axis=1)
            ## 注意现在暂时hollow的增强不需要了，但是如果后面还要的话要注意xyz轴的分辨率不一致，是各向异性的！！！！！！所以hollow不是一个球 是一个被压扁了的橄榄球，这样就不会因为z方向的原因导致半径太小，空心每次只有一点点大了
            hollow_coordinate = mask[distant <= radius].tolist()
            hollow_value = img.min()            
            img[:][hollow_coordinate[:,0]][hollow_coordinate[:,1]][hollow_coordinate[:,2]] = hollow_value
            return img,lab,spacing
                     
        
#class RandomBrightness_forWholeImg(AbstractTransform):
    ## 这个类 刘老师的RandomBrightness就是对全图调整亮度，但是刘老师那个还有把亮度变高的，对soma来说不用，只需要降低，所以改一下阈值就行，比如-self.dratio到0就行，而且刘老师这个对全图的变化是全图的，是线性变化，自己要改成非线性，可以先用类似上面CircleHollow进行球的局部变暗

class RandomBright_forLocalCircle(AbstractTransform):
    def __init__(self,p=0.7,dratio=0.1,multi_circle=True):
        super(RandomCircleHollow,self).__init__(p)
        assert dratio >= 0. and dratio < 1.
        self.dratio = dratio
        self.multi_circle = multi_circle
        #self.

    def __call__(self,img,lab,spacing):
        if np.random.random() < self.p:
            mask = np.argwhere(lab>0)
            bg = np.argwhere(lab==0)
            num = mask.shape[0] # shape[1]是3 对应了lab中大于0的元素的xyz坐标
            status = True
            #import ipdb;ipdb.set_trace()
            if multi_circle:
                num_circle = np.random.randint(5)### !!!问老师要不要改成圆柱，如果是圆柱的话 那足够大了 ，就没有必要多个了
            while status:
                #mask = np.argwhere(lab>0)
                #num = mask.shape[0] # shape是3 对应了lab中大于0的元素的xyz坐标
                centercir = mask[random.sample(range(num),k=1)] #从lab的mask中随机选一个点作为圆心
                #bg = np.argwhere(lab==0)
                radius_all = np.sum((centercir-bg)**2,axis=1)#如果直接检测lab的mask边缘的话会很麻烦，然后我们的半径肯定是小于选取的centercir(圆心)到边缘最小的点，但是边缘检测比较麻烦，且费时间，所以转换成求圆心与边缘外面那一圈背景的距离，与边缘紧挨着的那一圈背景的距离也就是，也就是圆心到背景最小的那几个距离。 而我们需要的是最小，所以这里也取最小，但是因为是边缘外侧的一圈，所以一定比圆心直接到边缘要大一点，但是最多就是大斜上方一个pixel的距离，也就是根号2，但是一般肯定不到根号2，也就是不会在斜上方，应该是在与边缘紧紧水平相邻的一个pixel，所以增大的距离肯定效于根号2，但是也是个大于1的根号，比较麻烦算，这里简单一点就用根号2替代
                radius = sorted(radius_all)[0] - 2 if sorted(radius_all)[0] - 2>0 else sorted(radius_all)[0] - 1 #但是不可能有根号2个pixel，所以我们就减去2，如果减去2小于0，那就减去1 ，如果减去1还小于零就证明取到了边缘上或者距离边缘只有1个pixel（这些都是不合适的），那就重新取
                if radius > 0:
                    status = False
            distant = np.sum((centercir-mask)**2,axis=1)
            radius = 50
            brightness_coordinate = mask[distant <= radius].tolist()
            mask_imgvales = img[lab>0]##感觉这里是rgb图的话，不太好保留RGB的三通道
            mm = mask_imgvales.max() - mask_imgvales.min()
            dmm = np.random.uniform(-self.dratio,0) * mm
            img[:][brightness_coordinate[:,0]][brightness_coordinate[:,1]][brightness_coordinate[:,2]] += dmm
            return img,lab,spacing


def random_crop_image_4D_forSoma(img, lab, spacing, target_shape):
    new_img = np.zeros((img.shape[0], *target_shape), dtype=img.dtype)
    for c in range(img.shape[0]):
        if c == 0:
            new_img[c],sz,sy,sx = image_util.random_crop_3D_image(img[c], target_shape)
        else:
            new_img[c] = img[sz:sz+target_shape[0], sy:sy+target_shape[1], sx:sx+target_shape[    2]]
    # processing the lab
    #import ipdb;ipdb.set_trace()
    if lab is not None:
        #import ipdb;ipdb.set_trace()
        new_lab = np.zeros(tuple(target_shape), dtype=lab.dtype)
        new_lab = lab[sz:sz+target_shape[0], sy:sy+target_shape[1], sx:sx+target_shape[2]]
        #new_lab,_,_,_ = image_util.random_crop_3D_image(lab,target_shape)
        return new_img, new_lab, spacing,sz,sy,sx
    return new_img, lab, spacing,sz,sy,sx
     
""" 
class RandomShift_forSoma(AbstractTransform):
    ## 这个老师的aug里面也说了 其实shift就是 crop + pad相结合的子集
    ## 所以用的时候直接 RandomCrop(p=... , crop_range=(0.95，1.0)) #需要注意这里crop_range一定要大（因为targrt_shape = shape*crop_range），因为我们shift几个pixel即可  然后接pad:这里用确定的pad 不用aug里面的RandomPaddng，而且因为spacing的问题，导致图之间尺寸不都是128，所以下面Padding的shape不好确定，所以shift和pad可以放到Scale类后面，要么就是写一个RandomShift 把RamdomCrop和确定的pad结合起来
    def __init__(self, p=0.5, imgshape=None, crop_range=(0.95, 1), per_axis=True, force_fg_sampling=False, pad_value=None):
        super(RandomCrop, self).__init__(p)
        self.imgshape = imgshape
        self.crop_range = crop_range
        self.per_axis = per_axis
        self.force_fg_sampling = force_fg_sampling

    def __call__(self, img, lab=None, spacing=None):
        if np.random.random() > self.p:
            return img, lab, spacing

        if self.crop_range[0] == self.crop_range[1]:
            target_shape = self.imgshape
            img, lab, spacing,sz,sy,sx = random_crop_image_4D_forSoma(img, lab, spacing, target_shape)
            return img, lab, spacing
        else:
            if self.force_fg_sampling: ##对soma来说，是想通过crop来实现shift，shift的尺度很小，所以crop不会出现没有前景的情况
                num_trail = 0
                while num_trail < 3:
                    shape, target_shape = get_random_shape(self.imgshape, self.crop_range, self.per_axis)
                    new_img, new_lab, new_spacing,sz,sy,sx = random_crop_image_4D_forSoma(img, lab, spacing, target_shape)
                    # check foreground existence
                    has_fg = False
                    if (new_lab > 0).any():
                        has_fg = True
                    if has_fg:
                        break

                    num_trail += 1
                else:
                    print("No foreground found after three random crops!")
            else:
                shape, target_shape = get_random_shape(self.imgshape, self.crop_range, self.per_axis)
                new_img, new_lab, new_spacing,sz,sy,sx = random_crop_image_4D_forSoma(img, lab, spacing, target_shape)

        if self.pad_value == None:
            pad_value = img.min()
        else:
            pad_value = self.pad_value
        
        shift_img = np.ones((img.shape[0],*target_shape),dtype=img.dtype) * pad_value
        #new_lab = np.zeros((*target_shape),dtype=lab.shape)
        sz = 
        sy = 
        sx = 
        for c in range(len(new_img)):
            new_img[c][sz:sz+shape[0], sy:sy+shape[1], sx:sx+shape[2]] = img
        if lab is not None:
            new_lab = np.zeros((*target_shape),dtype=lab.shape)
            new_lab[sz:sz+shape[0], sy:sy+shape[1], sx:sx+shape[2]] = lab


        
        
        
            return new_img, new_lab, new_spacing 
"""

def get_random_shape(img, scale_range, per_axis):
    #np.random.seed(200) 
    if type(img) == np.ndarray and img.size > 1024:
        shape = np.array(img[0].shape)
    else:
        shape = np.array(list(img))
    if per_axis:
        scales = np.random.uniform(*scale_range, size=len(shape))
    else:
        scales = np.array([np.random.uniform(*scale_range)] * len(shape))
    target_shape = np.round(shape * scales).astype(np.int)
    return shape, target_shape

def random_crop_image_4D(img, lab, spacing, target_shape):
    new_img = np.zeros((img.shape[0], *target_shape), dtype=img.dtype)
    for c in range(img.shape[0]):
        if c == 0:
            new_img[c],sz,sy,sx = image_util.random_crop_3D_image(img[c], target_shape)
        else:
            new_img[c] = img[sz:sz+target_shape[0], sy:sy+target_shape[1], sx:sx+target_shape[2]]
    # processing the lab
    #import ipdb;ipdb.set_trace()
    if lab is not None:
        #import ipdb;ipdb.set_trace()
        new_lab = np.zeros(tuple(target_shape), dtype=lab.dtype)
        new_lab = lab[sz:sz+target_shape[0], sy:sy+target_shape[1], sx:sx+target_shape[2    ]]
        #new_lab,_,_,_ = image_util.random_crop_3D_image(lab,target_shape)
        return new_img, new_lab, spacing
    return new_img, lab, spacing

class RandomCrop(AbstractTransform):
    def __init__(self, p=0.5, imgshape=None, crop_range=(0.85, 1), per_axis=True, force_fg_sampling=False):
        super(RandomCrop, self).__init__(p)
        self.imgshape = imgshape
        self.crop_range = crop_range
        self.per_axis = per_axis
        self.force_fg_sampling = force_fg_sampling

    def __call__(self, img, lab=None, spacing=None):
        if np.random.random() > self.p:
            return img, lab, spacing

        if self.crop_range[0] == self.crop_range[1]:
            target_shape = self.imgshape
            img, lab, spacing = random_crop_image_4D(img, lab, spacing, target_shape)
            return img, lab, spacing
        else:
            if self.force_fg_sampling:
                num_trail = 0
                while num_trail < 3:
                    shape, target_shape = get_random_shape(self.imgshape, self.crop_range, self.per_axis)
                    new_img, new_lab, new_spacing = random_crop_image_4D(img, lab, spacing, target_shape)
                    # check foreground existence
                    has_fg = False
                    if (new_lab > 0).any():
                        has_fg = True
                    if has_fg:
                        break

                    num_trail += 1
                else:
                    print("No foreground found after three random crops!")
            else:
                shape, target_shape = get_random_shape(self.imgshape, self.crop_range, self.per_axis)
                new_img, new_lab, new_spacing = random_crop_image_4D(img, lab, spacing, target_shape)

            return new_img, new_lab, new_spacing



class FixedPadding_forSoma(AbstractTransform):
    def __init__(self,ori_shape=(128,128,128),pad_value=None):
        super(AbstractTransform,self).__init__()
        self.ori_shape = ori_shape
        self.pad_value = pad_value
    def __call__(self,img,lab,spacing):
        if img.ndim == 3:
            img = img[None]
        shape = img.shape
        target_shape = self.ori_shape
        for si,ti in zip(shape,target_shape):
            assert si<=ti
        if self.pad_value == None:
            pad_value = img.min()
        else:
            pad_value = self.pad_value
        #import ipdb;ipdb.set_trace()
        
        new_img = np.ones((img.shape[0],*target_shape),dtype=img.dtype) * pad_value
        #new_lab = np.zeros((*target_shape),dtype=lab.shape)
        sz = 0
        sy = 0
        sx = 0
        for c in range(len(new_img)):
            new_img[c][sz:sz+shape[1], sy:sy+shape[2], sx:sx+shape[3]] = img[c]
        if lab is not None:
            new_lab = np.zeros(target_shape,dtype=lab.dtype)
            new_lab[sz:sz+shape[1], sy:sy+shape[2], sx:sx+shape[3]] = lab
            return new_img,new_lab,spacing
        else:
            return new_img,lab,spacing

class Random

"""
class Generate_multiSoma():
    def __init__(self,)

    def __call__()
"""


if __name__=="__main__":
    import SimpleITK as sitk
    #img = sitk.GetArrayFromImage(sitk.ReadImage("/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/data/task003_withNoRandomCrop_withTiffImageForRefer/ImgSoma_17302_00020-x_14992.3_y_21970.3_z_4344.8_img.tiff"))
    #img = sitk.GetArrayFromImage(sitk.ReadImage("/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/somaImage/soma_segmentation_training_images/17302_image_SEU200/ImgSoma_17302_00020-x_14992.3_y_21970.3_z_4344.8.tiff"))
    #lab = sitk.GetArrayFromImage(sitk.ReadImage("/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/somaImage/soma_segmentation_training_segmentations_mirrorImage/17302_seg_watershed_SEU200/seg_ImgSoma_17302_00020-x_14992.3_y_21970.3_z_4344.8.tiff"))
    img = sitk.GetArrayFromImage(sitk.ReadImage("/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/somaImage/soma_segmentation_training_images/17302_image_SEU200/ImgSoma_17302_00020-x_14992.3_y_21970.3_z_4344.8.tiff"))
    lab = sitk.GetArrayFromImage(sitk.ReadImage("/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/somaImage/soma_segmentation_training_segmentations_mirrorImage/17302_seg_watershed_SEU200/seg_ImgSoma_17302_00020-x_14992.3_y_21970.3_z_4344.8.tiff"))
    img = img[None]
    spacing = (1.0,0.23,0.23)
    imgshape = (128,128,128)
    t1 = RandomCrop(1.0,imgshape)
    img_c,lab_c,_ = t1(img,lab,spacing) 
    t2 = Padding_forSoma()
    img_p,lab_p,_ = t2(img_c,lab_c,spacing)
    img_p = np.squeeze(img_p)
    sitk.WriteImage(sitk.GetImageFromArray(img_p),"/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/testShift_img.tiff")
    sitk.WriteImage(sitk.GetImageFromArray(lab_p),"/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/testShift_lab.tiff")









    
