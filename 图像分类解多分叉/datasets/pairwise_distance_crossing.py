import numpy as np
import os
import time
from swc_handler import parse_swc, find_soma_node, get_child_dict, trim_out_of_box
from skimage.draw import line_nd
import SimpleITK as sitk
import random
from multiprocessing.pool import Pool

def get_soma_nearby_points(pos_dict, ignore_radius_from_soma, soma_pos):
    pts, remain_pts = [], []
    for idx, leaf in pos_dict.items():
        dist = np.linalg.norm(leaf[2] - soma_pos)
        if dist < ignore_radius_from_soma:
            pts.append(leaf[0])
        else:
            remain_pts.append(leaf[0])
    return set(pts), set(remain_pts)

def get_linkages_with_thresh(pos_dict, all_set, offspring_thresh):
    # get parents within thresh
    parent_dict = {}
    for idx in all_set:
        leaf = pos_dict[idx]
        # exclude parent
        os_id = 0
        cur_set = []
        while os_id < offspring_thresh:
            try:
                p_leaf = pos_dict[leaf[-1]]
                cur_set.append(p_leaf[0])
                
                leaf = p_leaf # update leaf
                os_id += 1
            except KeyError:
                break
        parent_dict[idx] = cur_set

    offspring_dict = {}
    for os, parents in parent_dict.items():
        for p_idx in parents:
            try:
                offspring_dict[p_idx].append(os)
            except KeyError:
                offspring_dict[p_idx] = [os]
    
    ## 下面这种不好，因为一个node的子节点不一定只有一个，下面这种只能选择其中一条路，可能漏掉一些path上的点
    """
    for idx in all_set:
        leaf = pos_dict[idx]
        p_idx = 0
        cur_set = []
        while p_idx < offspring_thresh:
            try:
                #import ipdb; ipdb.set_trace()
                c_leaf = child_dict[leaf[0]]
                cur_set.extend(c_leaf)
                
                leaf = pos_dict[c_leaf]
                p_id += 1
            except KeyError:
                break
        offspring_dict[idx] = cur_set
    """
    # convert to set
    for key, value in parent_dict.items():
        parent_dict[key] = set(value)
    for key, value in offspring_dict.items():
        offspring_dict[key] = set(value)

    return parent_dict, offspring_dict
    
def is_in_crop_box(x, y, z, cropsize):
    """
    cropsize must be in (z,y,x) order
    """
    if x < 0 or y < 0 or z < 0 or \
        x > cropsize[2] - 1 or \
        y > cropsize[1] - 1 or \
        z > cropsize[0] - 1:
        return False
    return True

def calc_pairwise_dist(output_dir, label_dir, outfile_prefix, ori_img, pos_dict, remain_set, exclude_set, offspring_thresh, only_calc_nearby_points=True):
    all_set = remain_set | exclude_set

    # get the linkages with thresh
    parent_dict, offspring_dict = get_linkages_with_thresh(pos_dict, all_set, offspring_thresh)
    
    t0 = time.time()
    nc = 0
    d1_idx_set = []
    d1_coord_set = []

    dmin_coord_listAll = []
    dmin_idx_listAll = []
    
    label_all = {}

    for idx in remain_set:
        if idx % 500 == 0:
            print(f"--> {nc / len(remain_set):.2%} finished in {time.time() - t0}s")
        leaf = pos_dict[idx]
        cur_set = set([idx]) | parent_dict[idx]
        try:
            cur_set = cur_set | offspring_dict[dict]
        except KeyError:
            pass
        pts = all_set - cur_set
        # all cur_distances
        cur_pos = leaf[2]
        cur_coords = []
        pts_arr = np.array(list(pts))
        for idx_ in pts_arr:
            cur_coords.append(pos_dict[idx_][2])
        # import ipdb; ipdb.set_trace()
        cur_coords = np.array(cur_coords)
        offset = cur_coords - cur_pos.reshape(1,-1)
        if only_calc_nearby_points: # speed up if no need to calculate far point.
            offset_max = np.fabs(offset).max(axis=1)
            offset_mask = offset_max < 5.0
            offset = offset[offset_mask]
            cur_coords = cur_coords[offset_mask]
            pts_arr  = pts_arr[offset_mask]
        dists = np.linalg.norm(offset, axis=1)
        # visual inspect nearby points
        visual = False
        if visual and not only_calc_nearby_points:
            for idx_, d, os in zip(pts_arr, dists, offset):
                if d < 2.0:
                    print(f"==> distance {d:.4f} and offset {os} for pair: {idx} / {idx_}")

        cur_dmin = 2.0
        dmin_coord_list = []
        dmin_idx_list = []
        for pos_, d, idx_ in zip(cur_coords, dists, pts_arr):
            if d <= 2.0:
                # remove points with common father
                has_common_parent = False
                for pr0 in parent_dict[idx]:
                    for pr1 in parent_dict[idx_]:
                        if pr0 == pr1:
                            has_common_parent = True
                            break
                    if has_common_parent == True:
                        break  # 这里这个根上面的break 是为了减少循环次数，提尕代码效率
                if not has_common_parent:
                    d1_coord_set.append((cur_pos, pos_))
                    d1_idx_set.append((idx,idx_))
                    if d < cur_dmin:
                        ### ！！！！！这样只找最小的感觉不是很恰当，因为一根fiber可能确实和好几根fiber都有crossing，所以一个点也就不止和另一个点离得很近，很可能是几个点
                        cur_dmin = d
                        dmin_coord_list = list((cur_pos, pos_))
                        dmin_idx_list = list((idx, idx_))
        
        # filtering by duplicate pair
        
        is_duplicate_pair = True  
        is_in_same_fiber = False
        #if len(dmin_coord_list) != 0:
            #import ipdb; ipdb.set_trace()
         
        if len(dmin_coord_list) != 0 and dmin_idx_list not in dmin_idx_listAll and dmin_idx_list[::-1] not in dmin_idx_listAll:
            #print("!!")
            is_duplicate_pair = False
            # filtering pairwise for the distance is too small (node in the same fiber(parent and offspring are 10 nodes))
            """ 
            for k in range(len(dmin_idx_listAll)):
                #import ipdb; ipdb.set_trace()
                idx, idx_ = dmin_idx_listAll[k]
                if dmin_idx_list[0] in parent_dict[idx] or dmin_idx_list[0] in offspring_dict[idx] or dmin_idx_list[0] in parent_dict[idx_] or dmin_idx_list[0] in offspring_dict[idx_] or dmin_idx_list[1] in parent_dict[idx] or dmin_idx_list[1] in offspring_dict[idx] or dmin_idx_list[1] in parent_dict[idx_] or dmin_idx_list[1] in offspring_dict[idx_]:
                    is_in_same_fiber = True
                    print("!!!")
                    break
            """

            """
            if not is_in_same_fiber:
                dmin_idx_listAll.append(dmin_idx_list)
                dmin_coord_listAll.append(dmin_coord_list)
            """
            
        nc += 1
        # extract crossing region
        if len(dmin_coord_list) != 0 and is_duplicate_pair == False and is_in_same_fiber == False:
        #if len(dmin_coord_list) != 0: 
            #print("!!!")
            dmin_idx = dmin_idx_list[0]
            dmin_idx_ = dmin_idx_list[1]

            #import ipdb; ipdb.set_trace()
            """ 
            if dmin_coord_list[0].all() == np.array((366.98,450.03,123.07)).all():
                import ipdb; ipdb.set_trace()
            """
            try:
                dmin_idxParent = list(parent_dict[dmin_idx])
                dmin_idx_parent = list(parent_dict[dmin_idx_])
                dmin_idxOffspring = list(offspring_dict[dmin_idx])
                dmin_idx_offspring = list(offspring_dict[dmin_idx_])
            except KeyError:
                break
            dmin_idxRegion = []
            dmin_idx_region = []

            """
            dmin_idxOffspring.extend(dmin_idxParent)
            dmin_idxOffspring.append(dmin_idx)
            dmin_idxRegion = dmin_idxOffspring
            dmin_idx_offspring.extend(dmin_idx_parent)
            dmin_idx_offspring.append(dmin_idx_)
            dmin_idx_region = dmin_idx_offspring
            
            dmin_idxRegion.sort()
            dmin_idx_region.sort()
            """

            dmin_idxRegion1 = []
                        
            dmin_idxRegion2 = []

            dmin_idxRegion3 = []
    
            dmin_idxRegion4 = []

            dmin_idxRegion5 = []

            dmin_idxRegion6 = []

            dmin_idxRegion.extend(dmin_idxParent)
            dmin_idxRegion.append(dmin_idx)
            dmin_idxRegion.extend(sorted(dmin_idxOffspring))
            dmin_idx_region.extend(dmin_idx_parent)
            dmin_idx_region.append(dmin_idx_)
            dmin_idx_region.extend(sorted(dmin_idx_offspring))

            #dmin_idxRegion.sort()
            #dmin_idx_region.sort()

            dmin_idxRegion1.extend(dmin_idxParent)
            dmin_idxRegion1.append(dmin_idx)
            dmin_idxRegion1.extend(sorted(dmin_idxOffspring))
            #dmin_idxRegion1.sort()

            dmin_idxRegion2.extend(dmin_idx_parent)
            dmin_idxRegion2.append(dmin_idx_)
            dmin_idxRegion2.extend(sorted(dmin_idx_offspring))
            #dmin_idxRegion2.sort()

            dmin_idxRegion3.extend(dmin_idxParent)
            dmin_idxRegion3.append(dmin_idx)
            dmin_idxRegion3.append(dmin_idx_)
            dmin_idxRegion3.extend(sorted(dmin_idx_parent,reverse=True))

            #import ipdb; ipdb.set_trace()
            dmin_idxRegion4.extend(dmin_idxParent)
            dmin_idxRegion4.append(dmin_idx)
            dmin_idxRegion4.append(dmin_idx_)
            dmin_idxRegion4.extend(sorted(dmin_idx_offspring))

            
            dmin_idxRegion5.extend(sorted(dmin_idxOffspring,reverse=True))
            dmin_idxRegion5.append(dmin_idx)
            dmin_idxRegion5.append(dmin_idx_)
            dmin_idxRegion5.extend(sorted(dmin_idx_parent,reverse=True))

            
            dmin_idxRegion6.extend(sorted(dmin_idxOffspring,reverse=True))
            dmin_idxRegion6.append(dmin_idx)
            dmin_idxRegion6.append(dmin_idx_)
            dmin_idxRegion6.extend(sorted(dmin_idx_offspring))


            img = np.zeros((32, 32,32),dtype=np.uint8)
 
            x_idx = round(pos_dict[idx][2][0])
            y_idx = round(pos_dict[idx][2][1])
            z_idx = round(pos_dict[idx][2][2])


            img1 = np.zeros((32, 32,32),dtype=np.uint8)
            xl1_idx,yl1_idx,zl1_idx = [], [], []

            for i in range(len(dmin_idxRegion1) - 1):
                #import ipdb;ipdb.set_trace()
                lin = line_nd(pos_dict[dmin_idxRegion1[i]][2][::-1], pos_dict[dmin_idxRegion1[i+1]][2][::-1],endpoint=True)

                xl1_idx.extend(list(lin[2]))
                yl1_idx.extend(list(lin[1]))
                zl1_idx.extend(list(lin[0]))


            xl1_idxArray = np.array(xl1_idx)
            yl1_idxArray = np.array(yl1_idx)
            zl1_idxArray = np.array(zl1_idx)

            xl1_idxArray -= (x_idx - 16)
            yl1_idxArray -= y_idx - 16
            zl1_idxArray -= z_idx - 16
            
            
            xn1_idx, yn1_idx, zn1_idx = [], [], []
            #import ipdb; ipdb.set_trace()
            for (xi, yi, zi) in zip(xl1_idxArray, yl1_idxArray, zl1_idxArray):
                if is_in_crop_box(xi, yi, zi, (32,32,32)):
                    xn1_idx.append(xi)
                    yn1_idx.append(yi)
                    zn1_idx.append(zi)
            
            
            img1[zn1_idx,yn1_idx,xn1_idx] = 255


            img2 = np.zeros((32, 32,32),dtype=np.uint8)
            xl2_idx,yl2_idx,zl2_idx = [], [], []

            for i in range(len(dmin_idxRegion2) - 1):
                #import ipdb;ipdb.set_trace()
                lin = line_nd(pos_dict[dmin_idxRegion2[i]][2][::-1], pos_dict[dmin_idxRegion2[i+1]][2][::-1],endpoint=True)

                xl2_idx.extend(list(lin[2]))
                yl2_idx.extend(list(lin[1]))
                zl2_idx.extend(list(lin[0]))


            xl2_idxArray = np.array(xl2_idx)
            yl2_idxArray = np.array(yl2_idx)
            zl2_idxArray = np.array(zl2_idx)

            xl2_idxArray -= (x_idx - 16)
            yl2_idxArray -= y_idx - 16
            zl2_idxArray -= z_idx - 16
            
            
            xn2_idx, yn2_idx, zn2_idx = [], [], []
            #import ipdb; ipdb.set_trace()
            for (xi, yi, zi) in zip(xl2_idxArray, yl2_idxArray, zl2_idxArray):
                if is_in_crop_box(xi, yi, zi, (32,32,32)):
                    xn2_idx.append(xi)
                    yn2_idx.append(yi)
                    zn2_idx.append(zi)
            
            
            img2[zn2_idx,yn2_idx,xn2_idx] = 255


            img3 = np.zeros((32, 32,32),dtype=np.uint8)
            xl3_idx,yl3_idx,zl3_idx = [], [], []

            for i in range(len(dmin_idxRegion3) - 1):
                #import ipdb;ipdb.set_trace()
                lin = line_nd(pos_dict[dmin_idxRegion3[i]][2][::-1], pos_dict[dmin_idxRegion3[i+1]][2][::-1],endpoint=True)

                xl3_idx.extend(list(lin[2]))
                yl3_idx.extend(list(lin[1]))
                zl3_idx.extend(list(lin[0]))


            xl3_idxArray = np.array(xl3_idx)
            yl3_idxArray = np.array(yl3_idx)
            zl3_idxArray = np.array(zl3_idx)

            xl3_idxArray -= (x_idx - 16)
            yl3_idxArray -= y_idx - 16
            zl3_idxArray -= z_idx - 16
            
            
            xn3_idx, yn3_idx, zn3_idx = [], [], []
            #import ipdb; ipdb.set_trace()
            for (xi, yi, zi) in zip(xl3_idxArray, yl3_idxArray, zl3_idxArray):
                if is_in_crop_box(xi, yi, zi, (32,32,32)):
                    xn3_idx.append(xi)
                    yn3_idx.append(yi)
                    zn3_idx.append(zi)
            
            
            img3[zn3_idx,yn3_idx,xn3_idx] = 255

            img4 = np.zeros((32, 32,32),dtype=np.uint8)
            xl4_idx,yl4_idx,zl4_idx = [], [], []

            for i in range(len(dmin_idxRegion4) - 1):
                #import ipdb;ipdb.set_trace()
                lin = line_nd(pos_dict[dmin_idxRegion4[i]][2][::-1], pos_dict[dmin_idxRegion4[i+1]][2][::-1],endpoint=True)

                xl4_idx.extend(list(lin[2]))
                yl4_idx.extend(list(lin[1]))
                zl4_idx.extend(list(lin[0]))


            xl4_idxArray = np.array(xl4_idx)
            yl4_idxArray = np.array(yl4_idx)
            zl4_idxArray = np.array(zl4_idx)

            xl4_idxArray -= (x_idx - 16)
            yl4_idxArray -= y_idx - 16
            zl4_idxArray -= z_idx - 16
            
            
            xn4_idx, yn4_idx, zn4_idx = [], [], []
            #import ipdb; ipdb.set_trace()
            for (xi, yi, zi) in zip(xl4_idxArray, yl4_idxArray, zl4_idxArray):
                if is_in_crop_box(xi, yi, zi, (32,32,32)):
                    xn4_idx.append(xi)
                    yn4_idx.append(yi)
                    zn4_idx.append(zi)
            
            
            img4[zn4_idx,yn4_idx,xn4_idx] = 255

            img5 = np.zeros((32, 32,32),dtype=np.uint8)
            xl5_idx,yl5_idx,zl5_idx = [], [], []

            for i in range(len(dmin_idxRegion5) - 1):
                #import ipdb;ipdb.set_trace()
                lin = line_nd(pos_dict[dmin_idxRegion5[i]][2][::-1], pos_dict[dmin_idxRegion5[i+1]][2][::-1],endpoint=True)

                xl5_idx.extend(list(lin[2]))
                yl5_idx.extend(list(lin[1]))
                zl5_idx.extend(list(lin[0]))


            xl5_idxArray = np.array(xl5_idx)
            yl5_idxArray = np.array(yl5_idx)
            zl5_idxArray = np.array(zl5_idx)

            xl5_idxArray -= (x_idx - 16)
            yl5_idxArray -= y_idx - 16
            zl5_idxArray -= z_idx - 16
            
            
            xn5_idx, yn5_idx, zn5_idx = [], [], []
            #import ipdb; ipdb.set_trace()
            for (xi, yi, zi) in zip(xl5_idxArray, yl5_idxArray, zl5_idxArray):
                if is_in_crop_box(xi, yi, zi, (32,32,32)):
                    xn5_idx.append(xi)
                    yn5_idx.append(yi)
                    zn5_idx.append(zi)
            
            
            img5[zn5_idx,yn5_idx,xn5_idx] = 255


            img6 = np.zeros((32, 32,32),dtype=np.uint8)
            xl6_idx,yl6_idx,zl6_idx = [], [], []

            for i in range(len(dmin_idxRegion6) - 1):
                #import ipdb;ipdb.set_trace()
                lin = line_nd(pos_dict[dmin_idxRegion6[i]][2][::-1], pos_dict[dmin_idxRegion6[i+1]][2][::-1],endpoint=True)

                xl6_idx.extend(list(lin[2]))
                yl6_idx.extend(list(lin[1]))
                zl6_idx.extend(list(lin[0]))


            xl6_idxArray = np.array(xl6_idx)
            yl6_idxArray = np.array(yl6_idx)
            zl6_idxArray = np.array(zl6_idx)

            xl6_idxArray -= (x_idx - 16)
            yl6_idxArray -= y_idx - 16
            zl6_idxArray -= z_idx - 16
            
            
            xn6_idx, yn6_idx, zn6_idx = [], [], []
            #import ipdb; ipdb.set_trace()
            for (xi, yi, zi) in zip(xl6_idxArray, yl6_idxArray, zl6_idxArray):
                if is_in_crop_box(xi, yi, zi, (32,32,32)):
                    xn6_idx.append(xi)
                    yn6_idx.append(yi)
                    zn6_idx.append(zi)
            
            
            img6[zn6_idx,yn6_idx,xn6_idx] = 255


            xl_idx,yl_idx,zl_idx = [], [], []

            for i in range(len(dmin_idxRegion) - 1):
                #import ipdb;ipdb.set_trace()
                lin = line_nd(pos_dict[dmin_idxRegion[i]][2][::-1], pos_dict[dmin_idxRegion[i+1]][2][::-1],endpoint=True)

                xl_idx.extend(list(lin[2]))
                yl_idx.extend(list(lin[1]))
                zl_idx.extend(list(lin[0]))


            xl_idxArray = np.array(xl_idx)
            yl_idxArray = np.array(yl_idx)
            zl_idxArray = np.array(zl_idx)

            xl_idxArray -= (x_idx - 16)
            yl_idxArray -= y_idx - 16
            zl_idxArray -= z_idx - 16
            
            #import ipdb;ipdb.set_trace()
            if z_idx<16 or z_idx > ori_img.shape[0]-16 or y_idx<16 or y_idx > ori_img.shape[1]-16 or x_idx<16 or x_idx > ori_img.shape[2]-16:
                #import ipdb; ipdb.set_trace()
                break; 
            ori_img_crop = ori_img[z_idx-16:z_idx+16,y_idx-16:y_idx+16,x_idx-16:x_idx+16]
            
            xn_idx, yn_idx, zn_idx = [], [], []
            #import ipdb; ipdb.set_trace()
            for (xi, yi, zi) in zip(xl_idxArray, yl_idxArray, zl_idxArray):
                if is_in_crop_box(xi, yi, zi, (32,32,32)):
                    xn_idx.append(xi)
                    yn_idx.append(yi)
                    zn_idx.append(zi)
            
            
            img[zn_idx,yn_idx,xn_idx] = 255
        
            xl_idx_,yl_idx_,zl_idx_ = [], [], []
            for i in range(len(dmin_idx_region) - 1):
                lin = line_nd(pos_dict[dmin_idx_region[i]][2][::-1], pos_dict[dmin_idx_region[i+1]][2][::-1], endpoint=True)

                xl_idx_.extend(list(lin[2]))
                yl_idx_.extend(list(lin[1]))
                zl_idx_.extend(list(lin[0]))


            xl_idx_array = np.array(xl_idx_)
            yl_idx_array = np.array(yl_idx_)
            zl_idx_array = np.array(zl_idx_)

            xl_idx_array -= x_idx - 16
            yl_idx_array -= y_idx - 16
            zl_idx_array -= z_idx - 16


            xn_idx_, yn_idx_, zn_idx_ = [], [], []
            for (xi, yi, zi) in zip(xl_idx_array, yl_idx_array, zl_idx_array):
                if is_in_crop_box(xi, yi, zi, (32, 32, 32)):
                    xn_idx_.append(xi)
                    yn_idx_.append(yi)
                    zn_idx_.append(zi)

            img[zn_idx_,yn_idx_,xn_idx_] = 255

            ## mip img
            img_mip = img.max(axis=0)
            ori_img_crop_mip = ori_img_crop.max(axis=0)
            ori_img_crop_mip = (ori_img_crop_mip - ori_img_crop_mip.mean()) / ori_img_crop_mip.std() * 255
            img1_mip = img1.max(axis=0)
            img2_mip = img2.max(axis=0)
            img3_mip = img3.max(axis=0)
            img4_mip = img4.max(axis=0)
            img5_mip = img5.max(axis=0)
            img6_mip = img6.max(axis=0)

            #import ipdb; ipdb.set_trace()
            #if np.sum(img!=0) != 0:
                #print("!!!")
            #import ipdb; ipdb.set_trace()
            ori_img_crop[img==0] = 0
            inp = np.expand_dims(ori_img_crop,axis=0)
            img1 = np.expand_dims(img1,axis=0)
            img2 = np.expand_dims(img2,axis=0)
            img3 = np.expand_dims(img3,axis=0)
            img4 = np.expand_dims(img4,axis=0)
            img5 = np.expand_dims(img5,axis=0)
            img6 = np.expand_dims(img6,axis=0) 

            img_allPaths = [img1,img2,img3,img4,img5,img6]
            label = np.array([1,1,0,0,0,0],dtype=np.uint8)
            index = [0,1,2,3,4,5]
            random.shuffle(index)
 
            #import ipdb;ipdb.set_trace()
            label = label[index]
            label_str = str()
            for l in label:
                label_str += str(l)
            
            """
            ## 感觉好像还是直接把标签直接存在图像的名字中更简单方便
            label_dict = {outfile_prefix:label}
            label_all.update(label_dict)
            """
            inp = np.concatenate((inp,img_allPaths[index[0]],img_allPaths[index[1]],img_allPaths[index[2]],img_allPaths[index[3]],img_allPaths[index[4]],img_allPaths[index[5]]),axis=0)

            if np.sum(img>0) != 0:
                dmin_idx_listAll.append(dmin_idx_list)
                dmin_coord_listAll.append(dmin_coord_list)

                prefix = str(pos_dict[idx][2][0]) + '_' + str(pos_dict[idx][2][1]) + '_' + str(pos_dict[idx][2][2]) + "_" + str(pos_dict[idx_][2][0]) + '_' + str(pos_dict[idx_][2][1]) + '_' + str(pos_dict[idx_][2][2])
                #import ipdb; ipdb.set_trace()
                

                #sitk.WriteImage(sitk.GetImageFromArray(ori_img_crop), output_dir +  '{}.tiff'.format(outfile_prefix + prefix + '_ori'))

                #sitk.WriteImage(sitk.GetImageFromArray(img), output_dir +  '{}.tiff'.format(outfile_prefix+ prefix))
                #import ipdb; ipdb.set_trace()
                import os
                if not os.path.exists(output_dir + '{}.tiff'.format(label_str + "_" + outfile_prefix + prefix + "inp")):
                    sitk.WriteImage(sitk.GetImageFromArray(inp), output_dir +  '{}.tiff'.format(label_str + "_" + outfile_prefix + prefix + "inp"))
                
                
                # save six path img to check the results
                #sitk.WriteImage(sitk.GetImageFromArray(img1), output_dir +  '{}.tiff'.format(outfile_prefix + prefix + '_img1'))
                #sitk.WriteImage(sitk.GetImageFromArray(img2), output_dir +  '{}.tiff'.format(outfile_prefix + prefix + '_img2'))
                #sitk.WriteImage(sitk.GetImageFromArray(img3), output_dir +  '{}.tiff'.format(outfile_prefix + prefix + '_img3'))
                #sitk.WriteImage(sitk.GetImageFromArray(img4), output_dir +  '{}.tiff'.format(outfile_prefix + prefix + '_img4'))
                #sitk.WriteImage(sitk.GetImageFromArray(img5), output_dir +  '{}.tiff'.format(outfile_prefix + prefix + '_img5'))
                #sitk.WriteImage(sitk.GetImageFromArray(img6), output_dir +  '{}.tiff'.format(outfile_prefix + prefix + '_img6'))

                # save six path mip img to check the results more convenient
                #import ipdb; ipdb.set_trace()
                #sitk.WriteImage(sitk.GetImageFromArray(img_mip),  output_dir + '{}.jpg'.format(outfile_prefix + prefix + '_imgMip'))
                #ori_img_crop_mip = ori_img_crop_mip.astype(np.uint8)
                #sitk.WriteImage(sitk.GetImageFromArray(ori_img_crop_mip),  output_dir + '{}.jpg'.format(outfile_prefix + prefix + '_oriImgMip'))

                #sitk.WriteImage(sitk.GetImageFromArray(img1_mip),  output_dir + '{}.jpg'.format(outfile_prefix + prefix + '_img1Mip'))
                #sitk.WriteImage(sitk.GetImageFromArray(img2_mip),  output_dir + '{}.jpg'.format(outfile_prefix + prefix + '_img2Mip'))
                #sitk.WriteImage(sitk.GetImageFromArray(img3_mip),  output_dir + '{}.jpg'.format(outfile_prefix + prefix + '_img3Mip'))
                #sitk.WriteImage(sitk.GetImageFromArray(img4_mip),  output_dir + '{}.jpg'.format(outfile_prefix + prefix + '_img4Mip'))
                #sitk.WriteImage(sitk.GetImageFromArray(img5_mip),  output_dir + '{}.jpg'.format(outfile_prefix + prefix + '_img5Mip'))
                #sitk.WriteImage(sitk.GetImageFromArray(img6_mip),  output_dir + '{}.jpg'.format(outfile_prefix + prefix + '_img6Mip'))             
            #np.savetxt(label_dir, label_all, fmt='%d')
    #return label_all

def pairwise_dist(swcfile, tifffile, output_dir, label_dir, outfile_prefix, ignore_radius_from_soma=50.0, offspring_thresh=10, imgshape=(256,512,512)):
    """
    Estimate pairwise distance of neurite pixels
    args:
    - ignore_radius_from_soma: radius nearby soma to ignored
    """
    print(f"Processing for swc: {swcfile}")

    tree = parse_swc(swcfile)
    ori_img = sitk.GetArrayFromImage(sitk.ReadImage(tifffile))
    # child_dict = get_child_dict(tree)
    pos_dict = {}
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p =leaf
        pos = np.array(list(map(float, (x,y,z))))
        pos_dict[idx] = (idx, type_, pos, r, p)
        
    soma_idx = find_soma_node(tree, p_soma=-1)
    soma_pos = pos_dict[soma_idx][2]
    # get the indices of soma-nearby points
    exclude_set, remain_set = get_soma_nearby_points(pos_dict, ignore_radius_from_soma, soma_pos)
    print(f"Number of exclude and remain points are: {len(exclude_set)}, {len(remain_set)}")

    # remove points with short offspring linkage
    calc_pairwise_dist(output_dir, label_dir, outfile_prefix, ori_img, pos_dict, remain_set, exclude_set, offspring_thresh, only_calc_nearby_points=True)
    #label_all = calc_pairwise_dist(output_dir, label_dir, outfile_prefix, ori_img, pos_dict, remain_set, exclude_set, offspring_thresh, only_calc_nearby_points=True)
    #return label_all

if __name__ == "__main__":
    import os, glob, sys
    #import ipdb; ipdb.set_trace()
    swc_dir = '/home/yjz/Projects/Auto_tracing/neuronet_new_0519_crossing_sixPathsModality/neuronet/data/swc/'
    tiff_dir = "/home/yjz/Projects/Auto_tracing/data/seu_mouse/crop_data/dendriteImageSecR_AllData/tiff/"
    output_dir = "/home/yjz/Projects/Auto_tracing/neuronet_new_0519_crossing_sixPathsModality/neuronet/data/task003_fixGenericDatasetBugs/train_input/"            
    label_dir = "/home/yjz/Projects/Auto_tracing/neuronet_new_0519_crossing_sixPathsModality/neuronet/data/task003_fixGenericDatasetBugs/label/label.txt"

    ignore_radius_from_soma = 50.
    offspring_thresh = 10
    num_threads = 8
    args_list = []
    #label_all = []
    for brain_dir in glob.glob(os.path.join(swc_dir,'*')):
        brain_id = brain_dir.split("/")[-1]
        for swcfile in glob.glob(os.path.join(brain_dir, '*swc')):
            #print(f"Processing for swc: {swcfile}")
            prefix = os.path.splitext(os.path.split(swcfile)[-1])[0]
            outfile_prefix = brain_id + '_' + prefix + '_'
            tifffile = tiff_dir + brain_id + "/" + "{}.tiff".format(prefix)
            #import ipdb; ipdb.set_trace()
            args = swcfile, tifffile, output_dir, label_dir, outfile_prefix, ignore_radius_from_soma, offspring_thresh
            args_list.append(args)
            #pairwise_dist(swcfile, tifffile, output_dir, label_dir, outfile_prefix, ignore_radius_from_soma, offspring_thresh)
            #labels = pairwise_dist(swcfile, tifffile, output_dir, label_dir, outfile_prefix, ignore_radius_from_soma, offspring_thresh)
            #label_all.extend(labels)

    """
    for args in args_list:
        #import ipdb; ipdb.set_trace()
        pairwise_dist(*args)
    """

    pt = Pool(num_threads)
    pt.starmap(pairwise_dist,args_list)
    pt.close()
    pt.join()
            
    #np.savetxt(label_dir, label_all, fmt='%d')

