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
    # convert to set
    for key, value in parent_dict.items():
        parent_dict[key] = set(value)
    for key, value in offspring_dict.items():
        offspring_dict[key] = set(value)

    return parent_dict, offspring_dict


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

        # check only very close points
        for pos_, d, idx_ in zip(cur_coords, dists, pts_arr):
            if d < 1.0:
                # remove points with common father
                has_common_parent = False
                for pr0 in parent_dict[idx]:
                    for pr1 in parent_dict[idx_]:
                        if pr0 == pr1:
                            has_common_parent = True
                if not has_common_parent:
                    d1_coord_set.append((cur_pos, pos_))
                    d1_idx_set.append((idx, idx_))
                

        if len(dists) == 0:
            dmin = 99.
        else:
            dmin = dists.min()
        if leaf[1] == 3 or leaf[1] == 4:
            dists_dend.append(dmin)
        elif leaf[1] == 2:
            dists_axon.append(dmin)

        nc += 1

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
