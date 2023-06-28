import glob
import os
import copy
import SimpleITK as sitk
import numpy as np
#from neuronet.augmentation.generic_augmentation_forSoma_someSpecificMethods import Padding_forSoma
from skimage import morphology

def affine_shift(img,lab,shift_z,shift_y,shift_x):
    D,H,W = img.shape
    
    #tem_img = img.copy()
    #tem_lab = lab.copy()
    #img = np.zeros((D+2,H+2,W+2),dtype=img.float32)
    #lab = np.zeros((D+2,H+2,W+2),dtype=lab.float32)
    #img[1:D+1,1:H+1,1:W+1] = temp_img
    #lab[1:D+1,1:H+1,1:W+1] = temp_lab

    new_img = np.zeros((D,H,W),dtype=img.dtype)
    new_lab = np.zeros((D,H,W),dtype=lab.dtype)

    y_new_temp = np.tile(np.arange(W), (H, 1))
    x_new_temp = np.arange(H).repeat(W).reshape(H, -1)
    #import ipdb;ipdb.set_trace()
    x_new = np.stack((x_new_temp,x_new_temp,x_new_temp)).astype(np.int)
    y_new = np.stack((y_new_temp,y_new_temp,y_new_temp)).astype(np.int)
    z_new = np.stack((np.ones((H,W))*0,np.ones((H,W))*1,np.ones((H,W))*2)).astype(np.int)
    
    x = x_new - shift_x
    y = y_new - shift_y
    z = z_new - shift_z

    # 避免目标图像对应的原图像中的坐标溢出    
    x = np.minimum(np.maximum(x, 0), H-1).astype(np.int)
    y = np.minimum(np.maximum(y, 0), W-1).astype(np.int)
    z = np.minimum(np.maximum(z, 0), D-1).astype(np.int)

    new_img[z_new,y_new,x_new] = img[z,y,x]
    new_lab[z_new,y_new,x_new] = lab[z,y,x]

    return new_img,new_lab

    

def generate(data_dir,img_folder,lab_folder):
    #import ipdb;ipdb.set_trace()
    for brain_dir in glob.glob(os.path.join(data_dir + img_folder,"*200")):
        brain_id = os.path.split(brain_dir)[-1][:5]
        lab_dir = os.path.join(data_dir,lab_folder,brain_id+"_seg_watershed_SEU200")
        img_files = glob.glob(os.path.join(brain_dir,"*"))
        for i,img_file in enumerate(img_files):
            prefix = os.path.splitext(os.path.split(img_file)[-1])[0]
            lab_prefix = "seg_" + prefix
            lab = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(lab_dir,lab_prefix+".tiff")))
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_file))
            mask_coordinate = np.argwhere(lab>0)
            #bg = np.argwhere(lab==0)
            center_cir = np.array(lab.shape) / 2
            distance_abs = abs(mask_coordinate - center_cir)
            #distance = np.sqrt(np.sum((mask_coordinate - center_cir)**2,axis=1))
            min_z_ori = max(distance_abs[:,0])## 这个最小距离的意思是：其他贴图进来的soma在z这个轴上，与原始图soma的原点的距离。  max()求得就是原始的这个soma的圆心到soma边缘，在z轴上的最大距离，因为其他贴图来的soma 只有大于这个最大距离才不会和原始的这个soma在空间上重叠。  
            
            #hypotenuse = np.sqrt(max(distance_abs[:,1])**2 + max(distance_abs[:,2])**2)
            min_y_ori = int(max(distance_abs[:,1])/3)#hypotenuse
            min_x_ori = int(max(distance_abs[:,2])/3)#hypotenuse
            #max_distance = sorted(distance,reverse=True)[0]
            #min_y_ori = max_distance
            #min_x_ori = max_distance
            img_addList = []
            lab_addList = []
            #import ipdb;ipdb.set_trace()
            num_multiSoma = np.random.randint(1,4) ##，这样的话最大取到3，也就是说在一个soma旁边最多再添加3个soma，一共最多4个soma
            for j in range(num_multiSoma):
                if i+j+1 > len(img_files)-1:
                    break
                img_addList.append(sitk.GetArrayFromImage(sitk.ReadImage(img_files[i+j+1])))
                prefix_add = os.path.splitext(os.path.split(img_files[i+j+1])[-1])[0]
                lab_prefix_add = "seg_" + prefix_add
                lab_addList.append(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(lab_dir,lab_prefix_add+".tiff"))))
            
            #new_img = np.zeros(img.shape,dtype=img.dtype)   
            new_img = copy.deepcopy(img)
            for k,img_add in enumerate(img_addList):
                lab_add = lab_addList[k]
                mask_coordinate_add = np.argwhere(lab_add>0)
                #bg_add = np.argwhere(lab_addList[k]==0)
                #center_cir = np.array(lab.shape) / 2
                distance_abs_add = abs(mask_coordinate_add - center_cir)
                #distance_add = np.sqrt(np.sum((mask_coordinate_add - center_cir)**2,axis=1))
            
                min_z_add = max(distance_abs_add[:,0])## 这里和上面同理，最终要取两个之间最大的那个，这样才能真正保证两个soma在空间上不重叠。   这个最小距离的意思是：其他贴图进来的soma在z这个轴上，与原始图soma的原点的距离。  max()求得就是原始的这个soma的圆心到soma边缘，在z轴上的最大距离，因为其他贴图来的soma 只有大于这个最大距离才不会和原始的这个soma在空间上重叠。  
                #hypotenuse_add = np.sqrt(max(distance_abs_add[:,1])**2 + max(distance_abs_add[:,2])**2)
                min_y_add = int(max(distance_abs[:,1])/3)#hypotenuse_add
                min_x_add = int(max(distance_abs[:,2])/3)#hypotenuse_add
                #max_distance_add = sorted(distance_add,reverse=True)[0]
                #min_y_add = max_distance_add
                #min_x_add = max_distance_add
                #import ipdb;ipdb.set_trace()
                min_z = max(min_z_ori,min_z_add)
                min_y = max(min_y_ori,min_y_add)
                min_x = max(min_x_ori,min_x_add) ##也就是说贴过去的图起码要大于这个距离
                if min_z*2 < (img.shape[0]/2-1*min_z):
                    shift_z = np.random.randint(min_z*2,img.shape[0]/2-1*min_z)##通过观察多个多soma的样本发现，最终移动的距离也不用太大，因为有不少样本是xy方向重叠很大，但是z方向分开的，所以z方向范围可以稍微大点，x y方向不用那么大就可以
                else:
                    shift_z = np.random.randint(img.shape[0]/2-1*min_z,min_z*2)
                if min_y+3 < (img.shape[1]-min_y*1):
                    shift_y = np.random.randint(min_y+3,img.shape[1]-min_y*1)
                else:
                    shift_y = np.random.randint(img.shape[1]-min_y*1,min_y+3)
                if min_x+3 < (img.shape[2]-min_x*1):
                    shift_x = np.random.randint(min_x+3,img.shape[2]-min_x*1)
                else:
                    shift_x = np.random.randint(img.shape[2]-min_x*1,min_x+3)

                
                random_z = np.random.choice([-1,1])
                random_y = np.random.choice([-1,1])
                random_x = np.random.choice([-1,1])
                ## ！！！！！！！！！！！！！ 下面这种方法太麻烦也可以做，但是要写八种对应不同的位置，我们利用仿射变换来做，更加简单
                """
                if random_z>0 and random_y>0 and random_x>0:
                    img_add = img_add[shift_z:,shift_y:,shift_x:]
                    lab_add = lab_add[shift_z:,shift_y:,shift_x:]
                    pad = Padding_forSoma()#！！！！！所以pad也不对 需要修改，还有相应的RandomShift那个类也有问题  需要修改  或者考虑一下直接修改RandomShift是不是不用这里了 两个改一个应该就行
                    spacing = (1.0,0.23,0.23)
                    #import ipdb;ipdb.set_trace()
                    img_add_p,lab_add_p,_ = pad(img_add,lab_add,spacing)
                    img_add_p = np.squeeze(img_add_p)
                """

                pad_value = img.min()
                img_add_p = np.ones(img.shape,dtype=img.dtype) * pad_value
                lab_add_p = np.zeros(lab.shape,dtype=lab.dtype)
                #import ipdb;ipdb.set_trace()
                if random_z>0 and random_y>0 and random_x>0:
                    img_add = img_add[shift_z:,shift_y:,shift_x:]
                    img_add_p[0:img_add.shape[0],0:img_add.shape[1],0:img_add.shape[2]] = img_add
                    lab_add = lab_add[shift_z:,shift_y:,shift_x:]
                    lab_add_p[0:lab_add.shape[0],0:lab_add.shape[1],0:lab_add.shape[2]] = lab_add

                elif random_z>0 and random_y>0 and random_x<0:
                    img_add = img_add[shift_z:,shift_y:,:-shift_x]
                    img_add_p[0:img_add.shape[0],0:img_add.shape[1],(img_add_p.shape[2]-img_add.shape[2]):] = img_add
                    lab_add = lab_add[shift_z:,shift_y:,:-shift_x]
                    lab_add_p[0:img_add.shape[0],0:img_add.shape[1]:,(lab_add_p.shape[2]-lab_add.shape[2]):] = lab_add

                elif random_z>0 and random_y<0 and random_x>0:
                    img_add = img_add[shift_z:,:-shift_y,shift_x:]
                    img_add_p[0:img_add.shape[0],(img_add_p.shape[1]-img_add.shape[1]):,0:img_add.shape[2]] = img_add
                    lab_add = lab_add[shift_z:,:-shift_y,shift_x:]
                    lab_add_p[0:img_add.shape[0],(img_add_p.shape[1]-img_add.shape[1]):,0:img_add.shape[2]] = lab_add

                elif random_z>0 and random_y<0 and random_x<0:
                    img_add = img_add[shift_z:,:-shift_y,:-shift_x]
                    img_add_p[0:img_add.shape[0],(img_add_p.shape[1]-img_add.shape[1]):,(img_add_p.shape[2]-img_add.shape[2]):] = img_add
                    lab_add = lab_add[shift_z:,:-shift_y,:-shift_x]
                    lab_add_p[0:img_add.shape[0],(img_add_p.shape[1]-img_add.shape[1]):,(img_add_p.shape[2]-img_add.shape[2]):] = lab_add

                elif random_z<0 and random_y>0 and random_x>0:
                    img_add = img_add[:-shift_z,shift_y:,shift_x:]
                    img_add_p[(img_add_p.shape[0]-img_add.shape[0]):,0:img_add.shape[1],0:img_add.shape[2]] = img_add
                    lab_add = lab_add[:-shift_z,shift_y:,shift_x:]
                    lab_add_p[(img_add_p.shape[0]-img_add.shape[0]):,0:img_add.shape[1],0:img_add.shape[2]] = lab_add

                elif random_z<0 and random_y>0 and random_x<0:
                    img_add = img_add[:-shift_z,shift_y:,:-shift_x]
                    img_add_p[(img_add_p.shape[0]-img_add.shape[0]):,0:img_add.shape[1],(img_add_p.shape[2]-img_add.shape[2]):] = img_add
                    lab_add = lab_add[:-shift_z,shift_y:,:-shift_x]
                    lab_add_p[(img_add_p.shape[0]-img_add.shape[0]):,0:img_add.shape[1],(img_add_p.shape[2]-img_add.shape[2]):] = lab_add
                elif random_z<0 and random_y<0 and random_x>0:
                    img_add = img_add[:-shift_z,:-shift_y,shift_x:]
                    img_add_p[(img_add_p.shape[0]-img_add.shape[0]):,(img_add_p.shape[1]-img_add.shape[1]):,0:img_add.shape[2]] = img_add
                    lab_add = lab_add[:-shift_z,:-shift_y,shift_x:]
                    lab_add_p[(img_add_p.shape[0]-img_add.shape[0]):,(img_add_p.shape[1]-img_add.shape[1]):,0:img_add.shape[2]] = lab_add

                elif random_z<0 and random_y<0 and random_x<0:
                    img_add = img_add[:-shift_z,:-shift_y,:-shift_x]
                    img_add_p[(img_add_p.shape[0]-img_add.shape[0]):,(img_add_p.shape[1]-img_add.shape[1]):,(img_add_p.shape[2]-img_add.shape[2]):] = img_add
                    lab_add = lab_add[:-shift_z,:-shift_y,:-shift_x]
                    lab_add_p[(img_add_p.shape[0]-img_add.shape[0]):,(img_add_p.shape[1]-img_add.shape[1]):,(img_add_p.shape[2]-img_add.shape[2]):] = lab_add
               
                #import ipdb;ipdb.set_trace() 
                #img_add_s,lab_add_s = affine_shift(img_add,lab_add,shift_z*random_z,shift_y*random_y,shift_x*random_x)
                selem_lab = np.ones((3,7,7),dtype=np.uint8)
                lab_add_dilation = morphology.dilation(lab_add_p,selem_lab)
                #sitk.WriteImage(sitk.GetImageFromArray(lab_add_p),"/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/lab_add_p.tiff")
                #sitk.WriteImage(sitk.GetImageFromArray(lab_add_dilation),"/home/yjz/Projects/Auto_tracing/neuronet_forSoma/neuronet/lab_add_dilation.tiff")
                #lab_add_dilation = morphology.dilation(lab_add_p,selem_lab)
                #import ipdb;ipdb.set_trace()
                img_add_p[lab_add_dilation==0] = 0
                #import ipdb;ipdb.set_trace()
                new_img[lab_add_dilation!=0] = 0
                new_img += img_add_p
            #sitk.WriteImage(sitk.GetImageFromArray(new_img),f"/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/somaImage/miniForTest/soma_segmentation_training_images/{brain_dir}_multiSoma/{prefix}_multiSoma.tiff")
            #sitk.WriteImage(sitk.GetImageFromArray(lab),f"/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/somaImage/miniForTest/soma_segmentation_training_images/{lab_dir}_multiSoma/{lab_prefix}_multiSoma.tiff")              
            #import ipdb; ipdb.set_trace()
            if not os.path.exists(f"{brain_dir}_multiSoma/{prefix}_multiSoma.tiff"):
                sitk.WriteImage(sitk.GetImageFromArray(new_img),f"{brain_dir}_multiSoma/{prefix}_multiSoma.tiff")
            if not os.path.exists(f"{lab_dir}_multiSoma/{lab_prefix}_multiSoma.tiff"):
                sitk.WriteImage(sitk.GetImageFromArray(lab),f"{lab_dir}_multiSoma/{lab_prefix}_multiSoma.tiff") 
                         
            #if not os.path.exists(f"{brain_dir}_multiSoma/{prefix}.tiff"):
                #sitk.WriteImage(sitk.GetImageFromArray(img),f"{brain_dir}_multiSoma/{prefix}.tiff")
            #if not os.path.exists(f"{lab_dir}_multiSoma/{lab_prefix}.tiff"):
                #sitk.WriteImage(sitk.GetImageFromArray(lab),f"{lab_dir}_multiSoma/{lab_prefix}.tiff")              
 


if __name__ == "__main__":
    data_dir = "/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/somaImage/miniForTest/"
    img_folder = "soma_segmentation_training_images"
    lab_folder = "soma_segmentation_training_segmentations_mirrorImage"
    generate(data_dir,img_folder,lab_folder)
