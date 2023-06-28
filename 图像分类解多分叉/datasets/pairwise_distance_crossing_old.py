#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : pairwise_distance.py
#   Author       : Yufeng Liu
#   Date         : 2021-05-28
#   Description  : 
#
#================================================================

import numpy as np
import time
from swc_handler import parse_swc, find_soma_node, get_child_dict, trim_out_of_box
from skimage.draw import line_nd
import SimpleITK as sitk
#import cv2

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
    #import ipdb;ipdb.set_trace()
    for os, parents in parent_dict.items():
        for p_idx in parents:
            try:
                offspring_dict[p_idx].append(os)
            except KeyError:
                offspring_dict[p_idx] = [os]
    # convert to set
    for key, value in parent_dict.items():
        parent_dict[key] = set(value)
    for key, value in offspring_dict.items():
        offspring_dict[key] = set(value)

    return parent_dict, offspring_dict

def is_in_crop_box(x,y,z,cropsize):
    """
    cropsize must be in (z,y,x) order
    """
    if x < 0 or y < 0 or z < 0 or \
        x > cropsize[2] - 1 or \
        y > cropsize[1] - 1 or \
        z > cropsize[0] - 1:
        return False
    return True

"""
def tps_cv2(source, target, img):
    #使用cv2自带的tps处理
    tps = cv2.createThinPlateSplineShapeTransformer()
    
    source_cv2 = source.reshape(1, -1, 2)
    target_cv2 = target.reshape(1, -1, 2)

    matches = list()
    for i in range(0, len(source_cv2[0])):
        matches.append(cv2.DMatch(i,i,0))

    tps.estimateTransformation(target_cv2, source_cv2, matches)
    new_img_cv2 = tps.warpImage(img)
    
    return new_img_cv2
"""

def calc_pairwise_dist(pos_dict, remain_set, exclude_set, offspring_thresh, only_calc_nearby_points=True):
    all_set = remain_set | exclude_set

    # get the linkages with thresh
    parent_dict, offspring_dict = get_linkages_with_thresh(pos_dict, all_set, offspring_thresh)

    dists_axon = []
    dists_dend = []
    t0 = time.time()
    nc = 0
    d1_idx_set = []
    d1_coord_set = []
    for idx in remain_set:
        if idx % 500 == 0:
            print(f'--> {nc / len(remain_set):.2%} finished in {time.time() - t0}s')

        leaf = pos_dict[idx]
        cur_set = set([idx]) | parent_dict[idx]
        try:
            cur_set = cur_set | offspring_dict[idx]
        except KeyError:
            pass
        pts = all_set - cur_set
        # all curr_distances
        cur_pos = leaf[2]
        cur_coords = []
        pts_arr = np.array(list(pts))
        for idx_ in pts_arr:
            cur_coords.append(pos_dict[idx_][2])
        cur_coords = np.array(cur_coords)
        #import ipdb; ipdb.set_trace()
        offset = cur_coords - cur_pos.reshape(1,-1)
        if only_calc_nearby_points: # speed up if no need to calcuate far points.
            offset_max = np.fabs(offset).max(axis=1)
            offset_mask = offset_max < 5.0
            offset = offset[offset_mask]
            cur_coords = cur_coords[offset_mask]
            pts_arr = pts_arr[offset_mask]
        dists = np.linalg.norm(offset, axis=1)
        # visual inspect nearby points
        visual = False
        if visual and not only_calc_nearby_points:
            for idx_, d, os in zip(pts_arr, dists, offset):
                if d < 2.0:
                    print(f'==> distance {d:.4f} and offset {os} for pair: {idx} / {idx_}')

        #import ipdb;ipdb.set_trace()
        # check only very close points
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
                if not has_common_parent:
                    d1_coord_set.append((cur_pos, pos_))
                    d1_idx_set.append((idx, idx_))
                    if d < cur_dmin:
                        cur_dmin = d
                        dmin_coord_list = list((cur_pos,pos_))
                        dmin_idx_list = list((idx,idx_))

        # extract crossing region
        if len(dmin_coord_list) != 0:
            dmin_idx = dmin_idx_list[0]
            dmin_idx_ = dmin_idx_list[1]

            dmin_idxParent = list(parent_dict[dmin_idx])
            dmin_idx_parent = list(parent_dict[dmin_idx_])
            dmin_idxOffspring = list(offspring_dict[dmin_idx])
            dmin_idx_offspring = list(offspring_dict[dmin_idx_])

            dmin_idxOffspring.extend(dmin_idxParent)
            dmin_idxOffspring.append(dmin_idx)
            dmin_idxRegion = dmin_idxOffspring
            dmin_idx_offspring.extend(dmin_idx_parent)
            dmin_idx_offspring.append(dmin_idx_)
            dmin_idx_region = dmin_idx_offspring

            dmin_idxRegion.sort()
            dmin_idx_region.sort()

            img = np.zeros((32, 32,32),dtype=np.uint8)

            x_idx = round(pos_dict[idx][2][0])
            y_idx = round(pos_dict[idx][2][1])
            z_idx = round(pos_dict[idx][2][2])

                        
            
            xl_idx,yl_idx,zl_idx = [], [], []
            termination = []
            #import ipdb; ipdb.set_trace()
            for i in range(len(dmin_idxRegion) - 1):
                """
                if i == 0:
                    lin = line_nd(pos_dict[idx][2][::-1], pos_dict[dmin_idxParent[i]][2][::-1],endpoint=True)
            
                    xl_idx.extend(list(lin[2]))
                    yl_idx.extend(list(lin[1]))
                    zl_idx.extend(list(lin[0]))
                else:
                """
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


            xn_idx, yn_idx, zn_idx = [], [], []  
            #import ipdb; ipdb.set_trace()
            for (xi, yi, zi) in zip(xl_idxArray, yl_idxArray, zl_idxArray):
                if is_in_crop_box(xi, yi, zi, (32,32,32)):
                    xn_idx.append(xi)
                    yn_idx.append(yi)
                    zn_idx.append(zi)          

            
            termination.extend([[xn_idx[0],yn_idx[0]],[xn_idx[-1],yn_idx[-1]]])
            #import ipdb; ipdb.set_trace()
            img[zn_idx,yn_idx,xn_idx] = 255
            
            xl_idx_,yl_idx_,zl_idx_ = [], [], []
            for i in range(len(dmin_idx_region) - 1):
                """
                if i == 0:
                    lin = line_nd(pos_dict[idx_][2][::-1], pos_dict[dmin_idx_parent[i]][2][::-1],endpoint=True)
            
                    xl_idx_.extend(list(lin[2]))
                    yl_idx_.extend(list(lin[1]))
                    zl_idx_.extend(list(lin[0]))
                else:
                """
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

            termination.extend([[xn_idx_[0],yn_idx_[0]],[xn_idx_[-1],yn_idx_[-1]]])

            img[zn_idx_,yn_idx_,xn_idx_] = 255
            ## generate nip image
            img_mip = img.max(axis=0)

            # generate trapezium from mip
            #import ipdb; ipdb.set_trace()
            """
            termination1_x = abs(round(pos_dict[dmin_idxRegion[0]][2][0]) - (x_idx - 16))
            termination1_y = abs(round(pos_dict[dmin_idxRegion[0]][2][1]) - (y_idx - 16))
            termination2_x = abs(round(pos_dict[dmin_idxRegion[-1]][2][0]) - (x_idx - 16))
            termination2_y = abs(round(pos_dict[dmin_idxRegion[-1]][2][1]) - (y_idx - 16))
            termination3_x = abs(round(pos_dict[dmin_idx_region[0]][2][0]) - (x_idx - 16))
            termination3_y = abs(round(pos_dict[dmin_idx_region[0]][2][1]) - (y_idx - 16))
            termination4_x = abs(round(pos_dict[dmin_idx_region[-1]][2][0]) - (x_idx - 16))
            termination4_y = abs(round(pos_dict[dmin_idx_region[-1]][2][1]) - (y_idx - 16))
            """
            
            termination1_x, termination1_y = sorted(termination,key=lambda termination:termination[0])[1]            
            termination2_x, termination2_y = sorted(termination,key=lambda termination:termination[0])[0]  
            termination3_x, termination3_y = sorted(termination,key=lambda termination:termination[0])[3] 
            termination4_x, termination4_y = sorted(termination,key=lambda termination:termination[0])[2] 
            
            source = np.array([[termination1_y,termination1_x],[termination2_y,termination2_x],[termination3_y,termination3_x],[termination4_y,termination4_x]])
            #print(source)
            target = np.array([[0,0],[0,31],[31,0],[31,31]])            
            #img_mip_resize = tps_cv2(source, target, img_mip)
            #import ipdb; ipdb.set_trace()
            prefix = str(pos_dict[idx][2][0]) + '_' + str(pos_dict[idx][2][1]) + '_' + str(pos_dict[idx][2][2])  
            sitk.WriteImage(sitk.GetImageFromArray(img), '{}.tiff'.format(prefix))
            sitk.WriteImage(sitk.GetImageFromArray(img_mip), '{}.jpg'.format(prefix))
            #sitk.WriteImage(sitk.GetImageFromArray(img_mip_resize), '{}_resize.jpg'.format(prefix))



            
                        

        
                

        """
        if len(dists) == 0:
            dmin = 99.
        else:
            dmin = dists.min()
        if leaf[1] == 3 or leaf[1] == 4:
            dists_dend.append(dmin)
        elif leaf[1] == 2:
            dists_axon.append(dmin)

        nc += 1
        """
    
    """
    # filtering by duplicate pair
    filtered_set = []
    filtered_idx_set = set()
    for i in range(len(d1_idx_set)):
        di = d1_idx_set[i]
        if (di[1], di[0]) not in filtered_idx_set:
            filtered_set.append(d1_coord_set[i])
            filtered_idx_set.add(di)

    print(f'{len(filtered_set)} left from {len(d1_idx_set)}')
    print(filtered_idx_set)

    # pairwise distance check
    npairs = 0
    ns = len(filtered_set)
    if ns > 1:
        for i in range(ns - 1):
            p01, p02 = filtered_set[i]
            has_near_pair = False
            for j in range(i+1, ns):
                p11, p12 = filtered_set[j]
                if (np.linalg.norm(p01 - p11) < 2.0 and np.linalg.norm(p02 - p12) < 2.0) or \
                    (np.linalg.norm(p01 - p12) < 2.0 and np.linalg.norm(p02 - p11) < 2.0):
                    has_near_pair = True
                    break
            if not has_near_pair:
                npairs += 1
    else:
        npairs = ns

    print(f"{npairs} crossing points within 1 voxel after filtering original {ns}")
    """ 
    
    return np.array(dists_dend), np.array(dists_axon)

def pairwise_dist(swcfile, ignore_radius_from_soma=50.0, offspring_thresh=10, imgshape=(256,512,512)):
    """
    Estimate pairwise distance of neurite pixels
    args:
    - ignore_radius_from_soma: radius nearby soma to ignored
    """
    tree = parse_swc(swcfile)
    #tree = trim_out_of_box(tree, imgshape)
    pos_dict = {}
    for i, leaf in enumerate(tree):
        idx, type_, x, y, z, r, p = leaf
        pos = np.array(list(map(float, (x,y,z))))
        pos_dict[idx] = (idx, type_, pos, r, p)

    soma_idx = find_soma_node(tree, p_soma=-1)
    soma_pos = pos_dict[soma_idx][2]
    #child_dict = get_child_dict(tree)
    # get the indices of soma-nearby points
    exclude_set, remain_set = get_soma_nearby_points(pos_dict, ignore_radius_from_soma, soma_pos)
    print(f'Number of exclude and remain points are: {len(exclude_set)}, {len(remain_set)}')

    # remove points with short offspring linkage
    dists_d, dists_a = calc_pairwise_dist(pos_dict, remain_set, exclude_set, offspring_thresh, only_calc_nearby_points=True)
    #import ipdb; ipdb.set_trace()
    print(f'Size of dendrite dists: {len(dists_d)}')
    print(f'Statis: {dists_d.mean():.4f}, {dists_d.std():.4f}, {dists_d.max():.4f}, {dists_d.min():.4f}')
    for d in (1.0, 2.0, 3.0, 4.0):
        nd = len(dists_d[dists_d < d])
        print(f'Number of points < {d:.1f} is {nd}')

    print(f'Size of axon dists: {len(dists_a)}')
    print(f'Statis: {dists_a.mean():.4f}, {dists_a.std():.4f}, {dists_a.max():.4f}, {dists_a.min():.4f}')
    for d in (1.0, 2.0, 3.0, 4.0):
        nd = len(dists_a[dists_a < d])
        print(f'Number of points < {d:.1f} is {nd}')

    return dists_d, dists_a


if __name__ == '__main__':
    import os, glob, sys

    swc_dir = './'
    ignore_radius_from_soma = 50.
    offspring_thresh = 10
    for brain_dir in glob.glob(os.path.join(swc_dir, '17109')):
        for swcfile in glob.glob(os.path.join(brain_dir, '*swc')):
            print(f'Processing for swc: {swcfile}')
            pairwise_dist(swcfile, ignore_radius_from_soma, offspring_thresh)
