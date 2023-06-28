#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : post_crossing.py
#   Author       : Yufeng Liu
#   Date         : 2021-12-12
#   Description  : 
#
#================================================================

import os, sys
import glob
import numpy as np


from swc_handler import parse_swc, parse_swc_upsample,scale_swc
from math_utils import calc_included_angles_from_coords, calc_included_angles_from_vectors
sys.path.append('../../src')
import morphology

def find_point_by_distance(pt, anchor_idx, is_parent, morph, dist, return_center_point=True, epsilon=1e-7):
    """
    Find the point of exact `dist` to the start pt on tree structure. args are:
    - pt: the start point, [coordinate]
    - anchor_idx: the first node on swc tree to trace, first child or parent node
    - is_parent: whether the anchor_idx is the parent of `pt`, otherwise child. 
                 if the node has several child, a random one is selected
    - morph: Morphology object for current tree
    - dist: distance threshold
    - return_center_point: whether to return the point with exact distance or
                 geometric point of all traced nodes
    - epsilon: small float to avoid zero-division error 
    """

    d = 0 
    ci = pt
    pts = [pt]
    while d < dist:
        try:
            cc = np.array(morph.pos_dict[anchor_idx][2:5])
        except KeyError:
            print(f"Parent/Child node not found within distance: {dist}")
            break
        d0 = np.linalg.norm(ci - cc) 
        d += d0
        if d < dist:
            ci = cc  # update coordinates
            pts.append(cc)

            if is_parent:
                anchor_idx = morph.pos_dict[anchor_idx][-1]
            else:
                if anchor_idx not in morph.child_dict:
                    break
                else:
                    anchor_idxs = morph.child_dict[anchor_idx]
                    anchor_idx = anchor_idxs[np.random.randint(0, len(anchor_idxs))]

    # interpolate to find the exact point
    dd = d - dist
    if dd < 0:
        pt_a = cc
    else:
        dcur = np.linalg.norm(cc - ci)
        assert(dcur - dd >= 0)
        pt_a = ci + (cc - ci) * (dcur - dd) / (dcur + epsilon)
        pts.append(pt_a)

    if return_center_point:
        pt_a = np.mean(pts, axis=0)

    return pt_a



class MultiFurcationAnalyzer(object):

    def __init__(self, swc_file, dist_thresh=1.5, line_length=5., soma_radius=10., downsampling=False, morph_existed=False, morph=None):
        if not morph_existed:
            try:
                tree = parse_swc(swc_file)
            except ValueError:
                tree = parse_swc_upsample(swc_file)
            if downsampling:
                tree = scale_swc(tree, 0.5)
        self.downsampling = downsampling

        self.swc_file = swc_file
        if not morph_existed:
            self.morph = morphology.Morphology(tree, p_soma=-1)
        else:
            self.morph = morph
        self.morph.get_critical_points()
        self.line_length = line_length
        self.soma_radius = soma_radius
        self.dist_thresh = dist_thresh


    def print_multifurcation(self):
        for mf in self.morph.multifurcation:
            print(f'multifurcation idx: {mf}')

    def calc_dist_furc(self):
        #print(f'Number of tips: {len(self.morph.tips)} in file: {self.swc_file}')
        dists = []
        pairs = []
        morph = self.morph

        cs = np.array(morph.pos_dict[morph.idx_soma][2:5])
        pset = set([])
        visited = set([])
        for tid in morph.tips:
            idx = tid
            pre_tip_id = None

            while idx != morph.idx_soma:
                if idx == -1:
                    break
                if idx in morph.child_dict:
                    n_child = len(morph.child_dict[idx])
                
                    if n_child > 1:
                        cur_tip_id = idx
                        if cur_tip_id in visited: 
                            idx = morph.pos_dict[idx][6]
                            break

                        if pre_tip_id is not None:
                            c0 = np.array(morph.pos_dict[cur_tip_id][2:5])
                            c1 = np.array(morph.pos_dict[pre_tip_id][2:5])
                            if np.linalg.norm(c0 - cs) < self.soma_radius:
                                break
                            dist = np.linalg.norm(c0 - c1)
                            if (dist < self.dist_thresh) and ((pre_tip_id, cur_tip_id) not in pset):
                                #print(f'{pre_tip_id}, {cur_tip_id}')
                                pairs.append((pre_tip_id, cur_tip_id, dist))
                                pset.add((pre_tip_id, cur_tip_id))
                            dists.append(dist)
                        #update tip
                        pre_tip_id = cur_tip_id
                        visited.add(pre_tip_id)
                
                idx = morph.pos_dict[idx][6]

        dists = np.array(dists)
        #print(f'Dist: {dists.mean():.2f}, {dists.std():.2f}, {dists.max():.2f}, {dists.min():.2f}')
        for pair in pairs:
            print(f'idx1 / idx2 and dist: {pair[0]} / {pair[1]} / {pair[2]}')        

        return dists, pairs

    def generate_crossing_ano_file(self):
        # create directory for current file
        swc_name = os.path.split(self.swc_file)[-1]
        prefix = os.path.splitext(swc_name)[0]
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        _, pairs = self.calc_dist_furc()
        # generate apo file
        apo_header = '##n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b'
        out_str = "f'{counter},  , {flag},{idx}, {z:.3f},{x:.3f},{y:.3f}, 0.000,0.000,0.000,0.000,0.000,,,,0,255,0'"

        apo_file = os.path.join(prefix, f'{prefix}.apo')
        with open(apo_file, 'w') as f1:
            f1.write(f'{apo_header}\n')
            counter = 0
            # for closing bifurcations
            flag = 'bifur2'
            for pair in pairs:
                idx = pair[0]
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(out_str) + '\n')
                counter += 1
                
                idx = pair[1]
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(out_str) + '\n')
                counter += 1

            # for mutlifurcation
            flag = 'trifur'
            for idx in self.morph.multifurcation:
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(out_str) + '\n')
                counter += 1

        # write ano file
        ano_file = os.path.join(prefix, f'{prefix}.ano')
        with open(ano_file, 'w') as f2:
            f2.write(f'APOFILE={os.path.split(apo_file)[-1]}\n')
            f2.write(f'SWCFILE={swc_name}\n')
        # copy the swc file to current folder
        os.system(f'cp {self.swc_file} {prefix}')


    ### WARNINING: Incomplete
    def calc_angle_furc(self):
        _, pairs = self.calc_dist_furc()

        included_angles = []
        for (idx1, idx2) in pairs:
            c1 = np.array(self.morph.pos_dict[idx1][2:5])
            c2 = np.array(self.morph.pos_dict[idx2][2:5])
            # children of idx1
            idx10, idx11 = self.morph.child_dict[idx1][:2]
            c3 = find_point_by_distance(c1, idx10, False, self.morph, self.line_length, False)
            c4 = find_point_by_distance(c1, idx11, False, self.morph, self.line_length, False)
            # father of idx2
            idx20 = self.morph.pos_dict[idx2][6]
            c0 = find_point_by_distance(c2, idx20, True, self.morph, self.line_length, False)
            # child of idx2
            pset = []
            idx = idx1
            while idx != idx2:
                pset.append(idx)
                idx = self.morph.pos_dict[idx][6]
            pset = set(pset)
            for idx21 in self.morph.child_dict[idx2]:
                if idx21 not in pset:
                    break
            c5 = find_point_by_distance(c2, idx21, False, self.morph, self.line_length, False)

            c12 = (c1 + c2) / 2.
            # estimate the angles
            coords10 = [c0] * 3
            anchor_coords = [c12] * 3
            coords11 = [c3, c4, c5]
            angs = calc_included_angles_from_coords(anchor_coords, coords10, coords11)
            included_angles.append(angs)

            # for 
            
        
    def calc_tip_statistics(self, dist_thresh=4.0):
        ntips = len(self.morph.tips)
        ntps = ntips * (ntips - 1) // 2
        
        ret = {}
        tip_list = list(self.morph.tips)
        for idx1, tip1 in enumerate(tip_list):
            c1 = np.array(self.morph.pos_dict[tip1][2:5])
            for idx2 in range(idx1+1, len(tip_list)):
                tip2 = tip_list[idx2]
                c2 = np.array(self.morph.pos_dict[tip2][2:5])
                # distance criterion
                dist = np.linalg.norm(c1 - c2)
                if dist > dist_thresh: continue
                
                #if self.swc_file == "/home/yjz/Project/Auto-tracing/data/seu_mouse/crop_data/dendriteImageMaxR_AllData/app2_upsample3_toyset_rotationAug_sorted/18454/10426_10338_6106.swc":
                    #import ipdb; ipdb.set_trace()
                # estimate the fiber distance
                pid1 = self.morph.pos_dict[tip1][6] # parent id for tip1
                if pid1 == -2:
                    continue
                pt1 = find_point_by_distance(c1, pid1, True, self.morph, self.line_length, False)
                pid2 = self.morph.pos_dict[tip2][6]
                if pid2 == -2:
                    continue
                pt2 = find_point_by_distance(c2, pid2, True, self.morph, self.line_length, False)
                # angle 
                v1 = (pt1 - c1).reshape((1,-1))
                v2 = (pt2 - c2).reshape((1,-1))
                ang = calc_included_angles_from_vectors(v1, v2)[0]
                ret[(tip1, tip2)] = (ang, dist)
                #print(tip1, tip2, ang, dist)

        return ret

    def generate_tip_error_ano_file(self, ang_thresh=75.0):
        # create directory for current file
        swc_name = os.path.split(self.swc_file)[-1]
        prefix = os.path.splitext(swc_name)[0]
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        ret = self.calc_tip_statistics()
        # filter by angle
        pairs = []
        for key, value in ret.items():
            if key[0] > ang_thresh:
                pairs.append(key)

        # generate apo file
        apo_header = '##n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b'
        out_str = "f'{counter},  , {flag},{idx}, {z:.3f},{x:.3f},{y:.3f}, 0.000,0.000,0.000,0.000,0.000,,,,0,255,0'"

        apo_file = os.path.join(prefix, f'{prefix}.apo')
        with open(apo_file, 'w') as f1: 
            f1.write(f'{apo_header}\n')
            counter = 0 
            # for closing bifurcations
            flag = 'break'
            for pair in pairs:
                idx = pair[0]
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(out_str) + '\n')
                counter += 1
    
                idx = pair[1]
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(out_str) + '\n')
                counter += 1


        # write ano file
        ano_file = os.path.join(prefix, f'{prefix}.ano')
        with open(ano_file, 'w') as f2: 
            f2.write(f'APOFILE={os.path.split(apo_file)[-1]}\n')
            f2.write(f'SWCFILE={swc_name}\n')
        # copy the swc file to current folder
        os.system(f'cp {self.swc_file} {prefix}')


    def generate_tip_and_crossing_ano_file(self, ang_thresh=75.0):
        # create directory for current file
        swc_name = os.path.split(self.swc_file)[-1]
        prefix = os.path.splitext(swc_name)[0]
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        ret = self.calc_tip_statistics()
        # filter by angle
        tip_pairs = []
        for key, value in ret.items():
            if key[0] > ang_thresh:
                tip_pairs.append(key)
        # crossing
        _, crossing_pairs = self.calc_dist_furc()

        if (len(crossing_pairs) == 0) and (len(tip_pairs) == 0) and (len(self.morph.multifurcation) == 0):
            return 
        

        # generate apo file
        apo_header = '##n,orderinfo,name,comment,z,x,y, pixmax,intensity,sdev,volsize,mass,,,, color_r,color_g,color_b'
        tip_str = "f'{counter},  , {flag},{idx}, {z:.3f},{x:.3f},{y:.3f}, 0.000,0.000,0.000,0.000,0.000,,,,255,255,0'"
        crossing_str = "f'{counter},  , {flag},{idx}, {z:.3f},{x:.3f},{y:.3f}, 0.000,0.000,0.000,0.000,0.000,,,,0,255,0'"

        apo_file = os.path.join(prefix, f'{prefix}.apo')
        with open(apo_file, 'w') as f1: 
            f1.write(f'{apo_header}\n')
            counter = 0 
            # crossing error
            flag = 'bifur2'
            for pair in crossing_pairs:
                idx = pair[0]
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(crossing_str) + '\n')
                counter += 1
                
                idx = pair[1]
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(crossing_str) + '\n')
                counter += 1

            # for mutlifurcation
            flag = 'trifur'
            for idx in self.morph.multifurcation:
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(crossing_str) + '\n')
                counter += 1


            # for closing bifurcations
            flag = 'break'
            for pair in tip_pairs:
                idx = pair[0]
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(tip_str) + '\n')
                counter += 1
    
                idx = pair[1]
                x,y,z = self.morph.pos_dict[idx][2:5]
                if self.downsampling:   # convert back
                    x *= 2
                    y *= 2
                    z *= 2
                f1.write(eval(tip_str) + '\n')
                counter += 1


        # write ano file
        ano_file = os.path.join(prefix, f'{prefix}.ano')
        with open(ano_file, 'w') as f2: 
            f2.write(f'APOFILE={os.path.split(apo_file)[-1]}\n')
            f2.write(f'SWCFILE={swc_name}\n')
        # copy the swc file to current folder
        os.system(f'cp {self.swc_file} {prefix}')


    def find_tip_and_crossing(self):
        """
        Check whether a pseudo-tip pair corresponds to a crossing-like structure
        """

        # helper functions
        def _is_tip_with_crossing(idx, cidxs, morph):
            while idx != morph.idx_soma:
                if idx == -1:
                    break
                if (idx in cidxs) or (idx in morph.multifurcation):
                    return True
                idx = morph.pos_dict[idx][6]
            return False
        ######### end of helper functions #########


        # get the tip pairs
        tps = self.calc_tip_statistics()    # dict: (tip1, tip2): (ang, dist)
        dfs = self.calc_dist_furc() # list: [(idx1, idx2, dist)]
        cidxs = set([idxs[1] for idxs in dfs[1]])
        num_found = 0
        num_error = 0
        for tp, value in tps.items():
            if value[0] > 90.0: # angle
                # check existance
                _is_found = _is_tip_with_crossing(tp[0], cidxs, self.morph)
                _is_found = _is_found | _is_tip_with_crossing(tp[1], cidxs, self.morph)
                num_found += _is_found
                num_error += _is_found
        print(f'{os.path.split(swc_file)[-1]}: {num_found} / {num_error} / {len(tps)}')
        return num_found, num_error, tps, dfs
        


if __name__ == '__main__':
    import pickle

    max_res = True
    is_gs = True
    if is_gs:
        #swc_dir = '/media/lyf/storage/seu_mouse/swc/xy1z1'
        swc_dir = "/home/yjz/Project/Auto-tracing/crossing/Myself_newMethod/data/app2/fused_tg0.0_alpha0.8_vanilla_bgMask0_upsample_to3pixel/00000"
        downsampling = True
    else:
        if not max_res:
            swc_dir = '/media/lyf/storage/seu_mouse/crop_data/processed/dendriteImageSecR/app2/'
            downsampling = False
        else:
            swc_dir = '/media/lyf/storage/seu_mouse/crop_data/processed/dendriteImageMaxR/app2/'
            downsampling = True
    
    outfile = 'tip_statistics.pkl'
    dist_thresh = 2.0
    line_length = 5.0
    soma_radius = 15.0

    # load all swc files
    swc_files = []
    if is_gs:
        for swc_file in glob.glob(os.path.join(swc_dir, '*[0-9].swc')):
            swc_files.append(swc_file)
    else:
        for swc_sub_dir in glob.glob(os.path.join(swc_dir, '*')):
            for swc_file in glob.glob(os.path.join(swc_sub_dir, '*[0-9].swc')):
                swc_files.append(swc_files)



    
    total_num_found, total_num_error, total_tps = 0, 0, 0
    for swc_file in swc_files:
        swc_name = os.path.split(swc_file)[-1]
        print(f'--> Processing for file: {swc_name}')
        try:
            mfa = MultiFurcationAnalyzer(swc_file, dist_thresh, line_length, soma_radius=soma_radius, downsampling=downsampling)
            if len(mfa.morph.tips) > 1000: continue
            mfa.generate_tip_and_crossing_ano_file()
            #num_found, num_error, tps, dfs = mfa.find_tip_and_crossing()
        except ValueError:
            print(f'----> Error for file {swc_name}')
            continue

        #total_num_found += num_found
        #total_num_error += num_error
        #total_tps += len(tps)
    #print(f'Total found pairs: {total_num_found} / {total_num_error} / {total_tps}\n')
    


    """
    #  crossing structure informations
    total_num_pairs, total_num_files = 0, 0
    for swc_file in swc_files:
        swc_name = os.path.split(swc_file)[-1]
        print(f'--> Processing for file: {swc_name}')
        try:
            mfa = MultiFurcationAnalyzer(swc_file, dist_thresh, line_length, downsampling=downsampling)
            if len(mfa.morph.tips) > 500: continue
            mfa.generate_crossing_ano_file()
        except ValueError:
            print(f'----> Error for file {swc_name}')
            continue

        #total_num_pairs += len(pairs)
        #total_num_files += 1
    print(f'Total found pairs: {total_num_pairs} / {total_num_files}\n')
    """

