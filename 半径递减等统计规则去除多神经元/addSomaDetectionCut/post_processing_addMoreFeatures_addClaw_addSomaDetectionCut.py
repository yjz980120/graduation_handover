import os, sys
import glob
import copy
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy

import SimpleITK as sitk

from swc_handler import parse_swc, parse_swc_upsample, find_soma_node, write_swc
import morphology
from post_crossing import MultiFurcationAnalyzer 


def angle_statistics_printInfo(angle_name, angle):
    print(f'{angle_name} overall statistics: {angle.mean():.2f}, {angle.std():.2f}, {angle.max():.2f}, {angle.min():.2f}')
    

def plot_hist(all_angles, img_name, out_img_dir):
    fig = sns.histplot(all_angles)
    hist_fig = fig.get_figure()
    hist_fig.savefig(os.path.join(out_img_dir, img_name))
    plt.close()


def hstack_list(all_angles_ps1,all_angles_ps2,all_angles_ps3,all_angles_s1s2,all_angles_s1s3,all_angles_s2s3,all_angles_ps_sum, all_angles_joint_max, all_angles_joint_s1s2_ps3, all_angles_joint_s2s3_ps1, all_angles_joint_s1s3_ps2):

    all_angles_ps1 = np.hstack(all_angles_ps1)
    all_angles_ps2 = np.hstack(all_angles_ps2)
    all_angles_ps3 = np.hstack(all_angles_ps3)
    all_angles_s1s2 = np.hstack(all_angles_s1s2)
    all_angles_s1s3 = np.hstack(all_angles_s1s3)
    all_angles_s2s3 = np.hstack(all_angles_s2s3)
    all_angles = np.hstack((all_angles_ps1,all_angles_ps2,all_angles_ps3,all_angles_s1s2,all_angles_s1s3,all_angles_s2s3))

    all_angles_ps_sum = np.hstack(all_angles_ps_sum)
    all_angles_joint_max = np.hstack(all_angles_joint_max)

    all_angles_joint_s1s2_ps3 = np.hstack(all_angles_joint_s1s2_ps3)
    all_angles_joint_s2s3_ps1 = np.hstack(all_angles_joint_s2s3_ps1)
    all_angles_joint_s1s3_ps2 = np.hstack(all_angles_joint_s1s3_ps2)

    return all_angles_ps1,all_angles_ps2,all_angles_ps3,all_angles_s1s2,all_angles_s1s3,all_angles_s2s3,all_angles,all_angles_ps_sum, all_angles_joint_max, all_angles_joint_s1s2_ps3, all_angles_joint_s2s3_ps1, all_angles_joint_s1s3_ps2
    

#def get_all_angles(all_angles_ps1,all_angles_ps2,all_angles_ps3,all_angles_s1s2,all_angles_s1s3,all_angles_s2s3, all_angles_ps_sum, all_angles_joint_max, all_angles_joint_s1s2_ps3, all_angles_joint_s2s3_ps1, all_angles_joint_s1s3_ps2, angles_ps1,angles_ps2,angles_ps3,angles_s1s2,angles_s1s3,angles_s2s3,angles_ps_sum, angles_joint_max, angles_joint_s1s2_ps3, angles_joint_s2s3_ps1, angles_joint_s1s3_ps2):
def get_all_angles(all_angles_joint_max,angles_joint_max):

    """
    all_angles_ps1.append(angles_ps1)
    all_angles_ps2.append(angles_ps2)
    all_angles_ps3.append(angles_ps3)
    all_angles_s1s2.append(angles_s1s2)
    all_angles_s1s3.append(angles_s1s3)
    all_angles_s2s3.append(angles_s2s3)

    all_angles_ps_sum.append(angles_ps_sum)
    """
    all_angles_joint_max.append(angles_joint_max)
    """
    all_angles_joint_s1s2_ps3.append(angles_joint_s1s2_ps3)
    all_angles_joint_s2s3_ps1.append(angles_joint_s2s3_ps1)
    all_angles_joint_s1s3_ps2.append(angles_joint_s1s3_ps2) 
    """

def find_points_forangles_multif(multif_cor,multif_idx,morph):
    anchor_idxP = morph.pos_dict[multif_idx][6]
    anchor_idx1 = morph.child_dict[multif_idx][0]
    anchor_idx2 = morph.child_dict[multif_idx][1]
    anchor_idx3 = morph.child_dict[multif_idx][2]
    point_p = find_point_with_distance(pt=multif_cor,anchor_idx=anchor_idxP, is_parent=True, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
    point_s1 = find_point_with_distance(pt=multif_cor,anchor_idx=anchor_idx1, is_parent=False, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
    point_s2 = find_point_with_distance(pt=multif_cor,anchor_idx=anchor_idx2, is_parent=False, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
    point_s3 = find_point_with_distance(pt=multif_cor,anchor_idx=anchor_idx3, is_parent=False, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
    return (point_p,anchor_idxP), (point_s1,anchor_idx1), (point_s2,anchor_idx2), (point_s3,anchor_idx3)


def find_multifucation(morph):
    morph.get_critical_points()
    multifurcation_cor, multifurcation_idx, multifurcation_radius = [], [], []
    multifurcation_num = 0#len(morph.multifurcation)
    soma_radius = 10
    soma_cor = np.array(morph.pos_dict[morph.idx_soma][2:5])
    for idx in morph.multifurcation:
        if np.linalg.norm(morph.pos_dict[idx][2:5]-soma_cor) >=soma_radius:
            multifurcation_idx.append(idx)
            multifurcation_radius.append(morph.pos_dict[idx][-2])
            multifurcation_cor.append(morph.pos_dict[idx][2:5])
            multifurcation_num += 1
    multifurcation_cor = np.array(multifurcation_cor)
    return multifurcation_cor, multifurcation_idx, multifurcation_radius, multifurcation_num
    

def find_points_forangles(start_point, cur_tip_id, pre_tip_id, morph):
    point_p = find_point_with_distance(pt=start_point, anchor_idx=cur_tip_id, is_parent=True, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
    anchor_idx1 = morph.child_dict[pre_tip_id][0]
    anchor_idx2 = morph.child_dict[pre_tip_id][1]
    #point_s1 = find_point_with_distance(pt=morph.pos_dict[pre_tip_id][2:5], anchor_idx=anchor_idx1, is_parent=False, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
    #point_s2 = find_point_with_distance(pt=morph.pos_dict[pre_tip_id][2:5], anchor_idx=anchor_idx2, is_parent=False, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
    point_s1 = find_point_with_distance(pt=start_point, anchor_idx=anchor_idx1, is_parent=False, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
    point_s2 = find_point_with_distance(pt=start_point, anchor_idx=anchor_idx2, is_parent=False, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
                          
    pset = []
    anchor_idx = pre_tip_id
    while anchor_idx != cur_tip_id:
        pset.append(anchor_idx)
        anchor_idx = morph.pos_dict[anchor_idx][6]
    pset = set(pset)
    for anchor_idx3 in morph.child_dict[cur_tip_id]:
        if anchor_idx3 not in pset:
            break
    #point_s3 = find_point_with_distance(pt=morph.pos_dict[pre_tip_id][2:5], anchor_idx=anchor_idx3, is_parent=False, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)
    point_s3 = find_point_with_distance(pt=start_point, anchor_idx=anchor_idx3, is_parent=False, morph=morph, dist=5, return_center_point=True, epsilon=1e-7)

    return (point_p,cur_tip_id),(point_s1,anchor_idx1),(point_s2,anchor_idx2),(point_s3,anchor_idx3)


def local_judge_radius_andIntensity_decreasing(morph, tiff_file, nearest_p, nearest_s1, nearest_s2, nearest_s3, count=5):
    ori_img = sitk.GetArrayFromImage(sitk.ReadImage(tiff_file))
    segment_p, segment_s1, segment_s2, segment_s3 = [nearest_p], [nearest_s1], [nearest_s2], [nearest_s3]
    cur_p, cur_s1, cur_s2, cur_s3 = nearest_p, nearest_s1, nearest_s2, nearest_s3
    count_p, count_s1, count_s2, count_s3 = 1, 1, 1, 1 
    decreasing_s1, decreasing_s2, decreasing_s3 = False, False, False

    segment_p_radius, segment_s1_radius, segment_s2_radius, segment_s3_radius = morph.pos_dict[cur_p][5], morph.pos_dict[cur_s1][5], morph.pos_dict[cur_s2][5], morph.pos_dict[cur_s3][5]

    segment_p_intensity, segment_s1_intensity, segment_s2_intensity, segment_s3_intensity = float(ori_img[round(morph.pos_dict[cur_p][4]), round(morph.pos_dict[cur_p][3]), round(morph.pos_dict[cur_p][2])]), float(ori_img[round(morph.pos_dict[cur_s1][4]), round(morph.pos_dict[cur_s1][3]), round(morph.pos_dict[cur_s1][2])]), float(ori_img[round(morph.pos_dict[cur_s2][4]), round(morph.pos_dict[cur_s2][3]), round(morph.pos_dict[cur_s2][2])]), float(ori_img[round(morph.pos_dict[cur_s3][4]), round(morph.pos_dict[cur_s3][3]), round(morph.pos_dict[cur_s3][2])])

    #import ipdb; ipdb.set_trace() 
    while cur_p != morph.idx_soma and count_p < count:
        cur_p = morph.pos_dict[cur_p][6]
        if cur_p not in morph.pos_dict:
            break
        count_p += 1
        segment_p.append(cur_p) 

        segment_p_radius += morph.pos_dict[cur_p][5]
        segment_p_intensity += ori_img[round(morph.pos_dict[cur_p][4]), round(morph.pos_dict[cur_p][3]), round(morph.pos_dict[cur_p][2])]
        
    while cur_s1 in morph.child_dict and len(morph.child_dict[cur_s1]) ==1 and count_s1 < count:
        cur_s1 = morph.pos_dict[cur_s1][6]
        if cur_s1 not in morph.pos_dict:
            break
        count_s1 += 1
        segment_s1.append(cur_s1)

        segment_s1_radius += morph.pos_dict[cur_s1][5]
        segment_s1_intensity += ori_img[round(morph.pos_dict[cur_s1][4]), round(morph.pos_dict[cur_s1][3]), round(morph.pos_dict[cur_s1][2])]

    while cur_s2 in morph.child_dict and len(morph.child_dict[cur_s2]) ==1 and count_s2 < count:
        cur_s2 = morph.pos_dict[cur_s2][6]
        if cur_s2 not in morph.pos_dict:
            break
        count_s2 += 1
        segment_s2.append(cur_s2)

        segment_s2_radius += morph.pos_dict[cur_s2][5]
        segment_s2_intensity += ori_img[round(morph.pos_dict[cur_s2][4]), round(morph.pos_dict[cur_s2][3]), round(morph.pos_dict[cur_s2][2])]

    while cur_s3 in morph.child_dict and len(morph.child_dict[cur_s3]) ==1 and count_s3 <= count:
        cur_s3 = morph.pos_dict[cur_s3][6]
        if cur_s3 not in morph.pos_dict:
            break
        count_s3 += 1
        segment_s3.append(cur_s3)

        segment_s3_radius += morph.pos_dict[cur_s3][5]
        segment_s3_intensity += ori_img[round(morph.pos_dict[cur_s3][4]), round(morph.pos_dict[cur_s3][3]), round(morph.pos_dict[cur_s3][2])]
    
    
    #import ipdb; ipdb.set_trace()
    if count_p != 0:
        segment_p_radius /= count_p
        segment_p_intensity /= count_p

    if count_s1 != 0: 
        segment_s1_radius /= count_s1
        segment_s1_intensity /= count_s1

    if count_s2 != 0:
        segment_s2_radius /= count_s2
        segment_s2_intensity /= count_s2

    if count_s3 != 0:
        segment_s3_radius /= count_s3
        segment_s3_intensity /= count_s3


    if segment_p_radius >= segment_s1_radius and segment_p_intensity >= segment_s1_intensity:
        decreasing_s1 = True
    
    if segment_p_radius >= segment_s2_radius and segment_p_intensity >= segment_s2_intensity:
        decreasing_s2 = True

    if segment_p_radius >= segment_s3_radius and segment_p_intensity >= segment_s3_intensity:
        decreasing_s3 = True
     
    return decreasing_s1, decreasing_s2, decreasing_s3 

                            

def calc_angles(center,p,nearest_p, s1,nearest_s1,s2,nearest_s2,s3,nearest_s3):
    """
    vec_p = center - p
    vec_s1 = center - s1
    vec_s2 = center - s2
    vec_s3 = center - s3
    """
    vec_p = p - center
    vec_s1 = s1 - center
    vec_s2 = s2 - center
    vec_s3 = s3 - center


    cos_ps1 = np.sum(vec_p * vec_s1) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_s1))
    angle_ps1 = math.acos(cos_ps1) / math.pi * 180
    cos_ps2 = np.sum(vec_p * vec_s2) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_s2))
    angle_ps2 = math.acos(cos_ps2) / math.pi * 180
    
    """
    ## 按照自己画的示意图， s1 s2点应该是固定，现在这种可能不固定，固定之后，ps2角肯定大于ps1????????????
    if angle_ps2< angle_ps1:
        angle_temp = angle_ps1
        angle_ps1 = angle_ps2
        angle_ps2 = angle_temp
        vec_temp = vec_s1
        vec_s1 = vec_s2
        vec_s2 = vec_temp
    """

    cos_ps3 = np.sum(vec_p * vec_s3) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_s3))
    angle_ps3 = math.acos(cos_ps3) / math.pi * 180

    cos_s1s2 = np.sum(vec_s2 * vec_s1) / (np.linalg.norm(vec_s2) * np.linalg.norm(vec_s1))
    angle_s1s2 = math.acos(cos_s1s2) / math.pi * 180
    cos_s1s3 = np.sum(vec_s1 * vec_s3) / (np.linalg.norm(vec_s1) * np.linalg.norm(vec_s3))
    angle_s1s3 = math.acos(cos_s1s3) / math.pi * 180
    cos_s2s3 = np.sum(vec_s2 * vec_s3) / (np.linalg.norm(vec_s2) * np.linalg.norm(vec_s3))
    angle_s2s3 = math.acos(cos_s2s3) / math.pi * 180

    angle_ps_sum = angle_ps1 + angle_ps2 + angle_ps3
    angle_joint_max = max((angle_s1s2+angle_ps3),(angle_s2s3+angle_ps1),(angle_s1s3+angle_ps2))
    if angle_joint_max == (angle_s1s2+angle_ps3):
        connection = (nearest_s1,nearest_s2)
        joint_max_angle2 = (angle_s1s2, angle_ps3)
    elif angle_joint_max == (angle_s2s3+angle_ps1):
        connection = (nearest_s2,nearest_s3)
        joint_max_angle2 = (angle_s2s3, angle_ps1)
    else:
        connection = (nearest_s1,nearest_s3)
        joint_max_angle2 = (angle_s1s3, angle_ps2)
    angle_joint_s1s2_ps3 = angle_s1s2+angle_ps3
    angle_joint_s2s3_ps1 = angle_s2s3+angle_ps1
    angle_joint_s1s3_ps2 = angle_s1s3+angle_ps2

    return angle_joint_max,joint_max_angle2, connection, (nearest_s1,nearest_s2, nearest_s3)
    

def calc_angles_withRadiusAndIntensity(center,p,nearest_p, s1,nearest_s1,s2,nearest_s2,s3,nearest_s3, decreasing_s1, decreasing_s2, decreasing_s3):
    """
    vec_p = center - p
    vec_s1 = center - s1
    vec_s2 = center - s2
    vec_s3 = center - s3
    """
    vec_p = p - center
    vec_s1 = s1 - center
    vec_s2 = s2 - center
    vec_s3 = s3 - center


    cos_ps1 = np.sum(vec_p * vec_s1) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_s1))
    angle_ps1 = math.acos(cos_ps1) / math.pi * 180
    cos_ps2 = np.sum(vec_p * vec_s2) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_s2))
    angle_ps2 = math.acos(cos_ps2) / math.pi * 180
    
    """
    ## 按照自己画的示意图， s1 s2点应该是固定，现在这种可能不固定，固定之后，ps2角肯定大于ps1????????????
    if angle_ps2< angle_ps1:
        angle_temp = angle_ps1
        angle_ps1 = angle_ps2
        angle_ps2 = angle_temp
        vec_temp = vec_s1
        vec_s1 = vec_s2
        vec_s2 = vec_temp
    """

    cos_ps3 = np.sum(vec_p * vec_s3) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_s3))
    angle_ps3 = math.acos(cos_ps3) / math.pi * 180

    cos_s1s2 = np.sum(vec_s2 * vec_s1) / (np.linalg.norm(vec_s2) * np.linalg.norm(vec_s1))
    angle_s1s2 = math.acos(cos_s1s2) / math.pi * 180
    cos_s1s3 = np.sum(vec_s1 * vec_s3) / (np.linalg.norm(vec_s1) * np.linalg.norm(vec_s3))
    angle_s1s3 = math.acos(cos_s1s3) / math.pi * 180
    cos_s2s3 = np.sum(vec_s2 * vec_s3) / (np.linalg.norm(vec_s2) * np.linalg.norm(vec_s3))
    angle_s2s3 = math.acos(cos_s2s3) / math.pi * 180

    angle_ps_sum = angle_ps1 + angle_ps2 + angle_ps3
    #angle_joint_max = max((angle_s1s2+angle_ps3),(angle_s2s3+angle_ps1),(angle_s1s3+angle_ps2))
    angle_joint = sorted([(angle_s1s2+angle_ps3),(angle_s2s3+angle_ps1),(angle_s1s3+angle_ps2)], reverse=True)
    angle_joint_max = angle_joint[0]
    angle_joint_sec = angle_joint[1]
    angle_joint_thi = angle_joint[2]

    #import ipdb; ipdb.set_trace()
    
    if angle_joint_max == (angle_s1s2+angle_ps3):
        if decreasing_s3:
            connection = (nearest_s1,nearest_s2)
            joint_max_angle2 = (angle_s1s2, angle_ps3)

        elif angle_joint_sec == (angle_s2s3+angle_ps1):
            if decreasing_s1:
                connection = (nearest_s2,nearest_s3)
                joint_max_angle2 = (angle_s2s3, angle_ps1)
            else:
                ## 这里不考虑decreasing_sX三个同时都是False的情况,因为三个同时为False的时候，就不会进入这个函数
                connection = (nearest_s1,nearest_s3)
                joint_max_angle2 = (angle_s1s3, angle_ps2)

        elif angle_joint_sec == (angle_s1s3+angle_ps2):
            if decreasing_s2:
                connection = (nearest_s1,nearest_s3)
                joint_max_angle2 = (angle_s1s3, angle_ps2)
            else:
                connection = (nearest_s2,nearest_s3)
                joint_max_angle2 = (angle_s2s3, angle_ps1)
                
    elif angle_joint_max == (angle_s2s3+angle_ps1):
        if decreasing_s1:
            connection = (nearest_s2,nearest_s3)
            joint_max_angle2 = (angle_s2s3, angle_ps1)

        elif angle_joint_sec == (angle_s1s2+angle_ps3):
            if decreasing_s3:
                connection = (nearest_s1,nearest_s2)
                joint_max_angle2 = (angle_s1s2, angle_ps3)
            else:
                connection = (nearest_s1,nearest_s3)
                joint_max_angle2 = (angle_s1s3, angle_ps2)

        elif angle_joint_sec == (angle_s1s3+angle_ps2):
            if decreasing_s2:
                connection = (nearest_s1,nearest_s3)
                joint_max_angle2 = (angle_s1s3, angle_ps2)
            else:
                connection = (nearest_s1,nearest_s2)
                joint_max_angle2 = (angle_s1s2, angle_ps3)

    else:
        if decreasing_s2:
            connection = (nearest_s1,nearest_s3)
            joint_max_angle2 = (angle_s1s3, angle_ps2)

        elif angle_joint_sec == (angle_s1s2+angle_ps3):
            if decreasing_s3:
                connection = (nearest_s1,nearest_s2)
                joint_max_angle2 = (angle_s1s2, angle_ps3)
            else:
                connection = (nearest_s2,nearest_s3)
                joint_max_angle2 = (angle_s2s3, angle_ps1)
        
        elif angle_joint_sec == (angle_s2s3+angle_ps1):
            if decreasing_s1:
                connection = (nearest_s2,nearest_s3)
                joint_max_angle2 = (angle_s1s3, angle_ps1)
            else:
                connection = (nearest_s1,nearest_s2)
                joint_max_angle2 = (angle_s1s2, angle_ps3)

    #angle_joint_s1s2_ps3 = angle_s1s2+angle_ps3
    #angle_joint_s2s3_ps1 = angle_s2s3+angle_ps1
    #angle_joint_s1s3_ps2 = angle_s1s3+angle_ps2

    return angle_joint_max,joint_max_angle2, connection, (nearest_s1,nearest_s2, nearest_s3)
    
def calc_angles_withClaw(center,p,nearest_p, s1,nearest_s1,s2,nearest_s2,s3,nearest_s3, claw_angle):
    """
    vec_p = center - p
    vec_s1 = center - s1
    vec_s2 = center - s2
    vec_s3 = center - s3
    """
    vec_p = p - center
    vec_s1 = s1 - center
    vec_s2 = s2 - center
    vec_s3 = s3 - center


    cos_ps1 = np.sum(vec_p * vec_s1) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_s1))
    angle_ps1 = math.acos(cos_ps1) / math.pi * 180
    cos_ps2 = np.sum(vec_p * vec_s2) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_s2))
    angle_ps2 = math.acos(cos_ps2) / math.pi * 180
    
    """
    ## 按照自己画的示意图， s1 s2点应该是固定，现在这种可能不固定，固定之后，ps2角肯定大于ps1????????????
    if angle_ps2< angle_ps1:
        angle_temp = angle_ps1
        angle_ps1 = angle_ps2
        angle_ps2 = angle_temp
        vec_temp = vec_s1
        vec_s1 = vec_s2
        vec_s2 = vec_temp
    """

    cos_ps3 = np.sum(vec_p * vec_s3) / (np.linalg.norm(vec_p) * np.linalg.norm(vec_s3))
    angle_ps3 = math.acos(cos_ps3) / math.pi * 180

    cos_s1s2 = np.sum(vec_s2 * vec_s1) / (np.linalg.norm(vec_s2) * np.linalg.norm(vec_s1))
    angle_s1s2 = math.acos(cos_s1s2) / math.pi * 180
    cos_s1s3 = np.sum(vec_s1 * vec_s3) / (np.linalg.norm(vec_s1) * np.linalg.norm(vec_s3))
    angle_s1s3 = math.acos(cos_s1s3) / math.pi * 180
    cos_s2s3 = np.sum(vec_s2 * vec_s3) / (np.linalg.norm(vec_s2) * np.linalg.norm(vec_s3))
    angle_s2s3 = math.acos(cos_s2s3) / math.pi * 180

    angle_ps_sum = angle_ps1 + angle_ps2 + angle_ps3
    angle_joint_max = max((angle_s1s2+angle_ps3),(angle_s2s3+angle_ps1),(angle_s1s3+angle_ps2))
    if angle_joint_max == (angle_s1s2+angle_ps3):
        connection = (nearest_s1,nearest_s2)
        joint_max_angle2 = (angle_s1s2, angle_ps3)
    elif angle_joint_max == (angle_s2s3+angle_ps1):
        connection = (nearest_s2,nearest_s3)
        joint_max_angle2 = (angle_s2s3, angle_ps1)
    else:
        connection = (nearest_s1,nearest_s3)
        joint_max_angle2 = (angle_s1s3, angle_ps2)
    angle_joint_s1s2_ps3 = angle_s1s2+angle_ps3
    angle_joint_s2s3_ps1 = angle_s2s3+angle_ps1
    angle_joint_s1s3_ps2 = angle_s1s3+angle_ps2

    if angle_ps1 > claw_angle and angle_ps2 > claw_angle and angle_ps3 > claw_angle:
        claw = True
    else:
        claw = False

    return angle_joint_max,joint_max_angle2, connection, (nearest_s1,nearest_s2, nearest_s3), claw
    

def find_point_with_distance(pt, anchor_idx, is_parent, morph, dist, return_center_point=True, epsilon=1e-7):
    """
    Find the point of exact `dist` to the start pt on tree structure. args are:
    - pt: the start point
    - anchor_idx: the first node on swc tree to trace. 
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
                anchor_idx = morph.pos_dict[anchor_idx][6]
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


def calc_dist_furc(swc_file, soma_radius, closing_bif_dist):
    try:
        tree = parse_swc(swc_file)
    except ValueError:
        tree = parse_swc_upsample(swc_file)
    if tree == []:
        print(f"{swc_file} may be error:blank! ")
        return np.array([0]),[0],[0],[0],[0],-1
    #print(swc_file)
    morph = morphology.Morphology(tree, p_soma=-1)
    morph.get_critical_points()

    #morph_a = morphology_angles.MorphAngles()

    #angs = morph_a.calc_outgrowth_angles(morph, indices_set = )

    print(f'Number of tips: {len(morph.tips)} in file: {swc_file}')
    if len(morph.tips) > 1000:
        return np.array([0]),[0],[0],[0],[0],-2

    dists = []
    pairs = []
    cur_tip_ids = []
    cur_tip_radius = []
    start_points = []
    soma_radius = soma_radius
    closing_bif_dist = closing_bif_dist
    num = 0
    soma_cor = np.array(morph.pos_dict[morph.idx_soma][2:5])
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
                    if pre_tip_id is not None:
                        c0 = np.array(morph.pos_dict[cur_tip_id][2:5])
                        c1 = np.array(morph.pos_dict[pre_tip_id][2:5])
                        if np.linalg.norm(c0 - soma_cor) < soma_radius:
                            break
                        dist = np.linalg.norm(c0 - c1)
                        if dist < closing_bif_dist and (cur_tip_id, pre_tip_id) not in pairs and (pre_tip_id, cur_tip_id) not in pairs:
                            num += 1
                            start_point = (np.array(morph.pos_dict[cur_tip_id][2:5]) + np.array(morph.pos_dict[pre_tip_id][2:5])) / 2
                            pairs.append((cur_tip_id, pre_tip_id))
                            cur_tip_ids.append(cur_tip_id)
                            cur_tip_radius.append(morph.pos_dict[cur_tip_id][-2])
                            start_points.append(start_point)
                            
                    #update tip
                    pre_tip_id = cur_tip_id
            
            idx = morph.pos_dict[idx][6]

    dists = np.array(dists)
    start_points = np.array(start_points)

    return start_points, pairs, cur_tip_ids, cur_tip_radius, morph, num

def calc_dist_furc_addSomaDetCut(morph, swc_file, soma_radius, closing_bif_dist):
    print(f'Number of tips: {len(morph.tips)} in file: {swc_file}')
    if len(morph.tips) > 1000:
        return np.array([0]),[0],[0],[0],[0],-2

    dists = []
    pairs = []
    cur_tip_ids = []
    cur_tip_radius = []
    start_points = []
    soma_radius = soma_radius
    closing_bif_dist = closing_bif_dist
    num = 0
    soma_cor = np.array(morph.pos_dict[morph.idx_soma][2:5])
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
                    if pre_tip_id is not None:
                        c0 = np.array(morph.pos_dict[cur_tip_id][2:5])
                        c1 = np.array(morph.pos_dict[pre_tip_id][2:5])
                        if np.linalg.norm(c0 - soma_cor) < soma_radius:
                            break
                        dist = np.linalg.norm(c0 - c1)
                        if dist < closing_bif_dist and (cur_tip_id, pre_tip_id) not in pairs and (pre_tip_id, cur_tip_id) not in pairs:
                            num += 1
                            start_point = (np.array(morph.pos_dict[cur_tip_id][2:5]) + np.array(morph.pos_dict[pre_tip_id][2:5])) / 2
                            pairs.append((cur_tip_id, pre_tip_id))
                            cur_tip_ids.append(cur_tip_id)
                            cur_tip_radius.append(morph.pos_dict[cur_tip_id][-2])
                            start_points.append(start_point)
                            
                    #update tip
                    pre_tip_id = cur_tip_id
            
            idx = morph.pos_dict[idx][6]

    dists = np.array(dists)
    start_points = np.array(start_points)

    return start_points, pairs, cur_tip_ids, cur_tip_radius, morph, num



def exclude_main_branch(crossing_likely_cor, crossing_likely_idx, crossing_likely_pairs_andIdx, crossing_likely_radius, morph, soma_threshold=12.0, radius_threshold=10.5, topo_threshshold=3):
    ## default: soma_threshold=12.0, radius_threshold=2.5, topo_threshshold=3
    crossing_likely_radius = np.array(crossing_likely_radius)
    include_crossing_likely_cor = []
    include_crossing_likely_idx = []
    include_crossing_likely_pairs_idx = []
    soma_cor = np.array(morph.pos_dict[morph.idx_soma][2:5])
    ## x ,y ,z 方向分辨率不同，总体上来说差不多接近1：1：0.3  所以可以对Z方向进行一定的加权，这样算的更准一些
    soma_cor *= np.array([1,1,3]) 
    ## 首先soma附近的一定保留：
    crossing_likely_cor_weight = copy.deepcopy(crossing_likely_cor)
    crossing_likely_cor_weight *= np.array([1,1,3])
    dist = np.linalg.norm((soma_cor - crossing_likely_cor),axis=1) 
    include_id_soma = np.argwhere(dist > soma_threshold)
    include_id_radius = np.argwhere(crossing_likely_radius <= radius_threshold)
    include_id_soma = set(include_id_soma.reshape(include_id_soma.shape[0]).tolist())
    include_id_radius = set(include_id_radius.reshape(include_id_radius.shape[0]).tolist())
    
    #include_id = include_id_soma
    
    #include_id = np.vstack((include_id,include_id_radius))
    #include_id = include_id.reshape(include_id.shape[0]).tolist()
    #include_id = set(include_id)
    include_id = include_id_soma & include_id_radius
    
    furcation = []
    furcation.extend(morph.multifurcation)
    furcation.extend(morph.bifurcation)
    furcation = set(furcation)
    
    exclude_num = len(crossing_likely_cor) - len(include_id)
    ## 注意include_id 先去个重
    for i in include_id:
        include_crossing_likely_cor.append(crossing_likely_cor[i])
        include_crossing_likely_idx.append(crossing_likely_idx[i])
        include_crossing_likely_pairs_idx.append(crossing_likely_pairs_andIdx[i])


     
    exclude_topo_id = []
    for i, idx in enumerate(include_crossing_likely_idx):
        cur_idx = idx
        cur_idx_topoRank = 1
        while cur_idx != morph.idx_soma:
            cur_idx = morph.pos_dict[cur_idx][6]
            ## soma里的topo层级不算
            if np.linalg.norm((soma_cor - np.array(morph.pos_dict[cur_idx][2:5]))) < soma_threshold:
                break
            if cur_idx in furcation:
                cur_idx_topoRank += 1
            ## 已经超过四级就不用算了
            if cur_idx_topoRank > topo_threshshold:
                break
        if cur_idx_topoRank <= topo_threshshold:
            exclude_num += 1
            exclude_topo_id.append(i)
    slow = 0
    fast = 0
    while fast < len(include_crossing_likely_cor):
        if exclude_topo_id and fast != exclude_topo_id[0]:
            include_crossing_likely_idx[slow] = include_crossing_likely_idx[fast]
            include_crossing_likely_pairs_idx[slow] = include_crossing_likely_pairs_idx[fast]
            include_crossing_likely_cor[slow] = include_crossing_likely_cor[fast]
            slow += 1
        elif exclude_topo_id and fast == exclude_topo_id[0]:
            exclude_topo_id = exclude_topo_id[1:]
        
        fast += 1

    if slow:
        include_crossing_likely_idx = include_crossing_likely_idx[:slow]
        include_crossing_likely_pairs_idx = include_crossing_likely_pairs_idx[:slow]
        include_crossing_likely_cor = include_crossing_likely_cor[:slow]
    

    return include_crossing_likely_idx, include_crossing_likely_pairs_idx, include_crossing_likely_cor, exclude_num

def delete_shortlySegment(morph, crossing_likely_pairs_idx, length=6.0):
    post_morph = copy.deepcopy(morph)
    shortlySegment_num = 0
    for idx in crossing_likely_pairs_idx:
        if type(idx) == tuple:
            ## 那就是邻近二分叉
            s1 = post_morph.child_dict[idx[1]][0]
            s2 = post_morph.child_dict[idx[1]][1]
            tip_id = idx[1]
            pset = []
            while tip_id != idx[0]:
                pset.append(tip_id)
                tip_id = post_morph.pos_dict[tip_id][6]

            pset = set(pset)
            for s3 in post_morph.child_dict[idx[0]]:
                if s3 not in pset:
                    break
        else:
            ## 那就是正常的三分叉 
            s1 = post_morph.child_dict[idx][0]
            s2 = post_morph.child_dict[idx][1]
            s3 = post_morph.child_dict[idx][2]

        ss = [s1,s2,s3]
        for s in ss:
            dist = 0
            shortlySegment = False
            try:
                while post_morph.child_dict[s]:
                    s_son = post_morph.child_dict[s]
                    if len(s_son) != 1:
                        break
                    dist += np.linalg.norm((np.array(post_morph.pos_dict[s][2:5]) - np.array(post_morph.pos_dict[s_son[0]][2:5])))
                    if dist > length:
                        break
                    s = s_son[0]
            except KeyError:
                ## 此时没有子节点了，也就是到末端了
                shortlySegment = True
            if shortlySegment:
                temp = list(post_morph.pos_dict[s])
                temp[6] = -2
                post_morph.pos_dict[s] = tuple(temp)
                shortlySegment_num += 1
    
    print(f"There are {shortlySegment_num} shortly segments!")
    return post_morph, shortlySegment_num
            

def find_breakTip2Pairs(swc_file, dist_thresh, line_length, angle_thresh):
    mfa = MultiFurcationAnalyzer(app2_swc_file, dist_thresh, line_length, soma_radius=soma_threshold, downsampling=False)
    ret = mfa.calc_tip_statistics()
    break_tip2_pairs = [] # 这里存的就是那一对断掉的末端点
    for key, value in ret.items():
        if value[0] > angle_thresh:
            break_tip2_pairs.append(key)

    return break_tip2_pairs

def find_breakTip2Pairs_somaDetCut(swc_file, morph, dist_thresh, line_length, angle_thresh):
    mfa = MultiFurcationAnalyzer(app2_swc_file, dist_thresh, line_length, soma_radius=soma_threshold, downsampling=False)
    ret = mfa.calc_tip_statistics()
    break_tip2_pairs = [] # 这里存的就是那一对断掉的末端点
    for key, value in ret.items():
        if value[0] > angle_thresh and key in morph.pos_dict:
            break_tip2_pairs.append(key)

    return break_tip2_pairs
  
    
"""        
def pair_breakTip2_crossinglikely(morph, break_tip2_pairs, include_crossing_likely_idx):
    pair_breakTip2_crossinglikely_idx = []
    pair_num = 0
    for tip1, tip2 in break_tip2_pairs:
        cur_node1 = tip1
        cur_node2 = tip2
        in_include_idx = False
        while cur_node1 != -1 and cur_node1 != -2:
            if cur_node1 in include_crossing_likely_idx:
                pair_num += 1
                pair_breakTip2_crossinglikely_idx.append(cur_node1)
                in_include_idx = True
                break
            cur_node1 = morph.pos_dict[cur_node1][6]
        if not in_include_idx:
            while cur_node2 != -1 and cur_node2 != -2:
                if cur_node2 in include_crossing_likely_idx:
                    pair_num += 1
                    pair_breakTip2_crossinglikely_idx.append(cur_node2)
                    break
                cur_node2 = morph.pos_dict[cur_node2][6] 

    pair_breakTip2_crossinglikely_idx = set(pair_breakTip2_crossinglikely_idx)
    print(f"There are {pair_num} break_tip2 pair crossinglikely, and breakTip2 match {len(pair_breakTip2_crossinglikely_idx)} crossinglikely !")
    return pair_breakTip2_crossinglikely_idx, pair_num
"""

def pair_breakTip2_crossinglikely(morph, break_tip2_pairs, include_crossing_likely_idx):
    pair_breakTip2_crossinglikely_idx = []
    break_pair_2cross_idx = []
    pair_num = 0
    for tip1, tip2 in break_tip2_pairs:
        cur_node1 = tip1
        cur_node2 = tip2
        in_include_idx = False
        cur_node1_pair = False
        cur_node2_pair = False

        while cur_node1 != -1 and cur_node1 != -2:
            if cur_node1 in include_crossing_likely_idx:
                cur_node1_pair = True
                pair_num += 1
                pair_breakTip2_crossinglikely_idx.append(cur_node1)
                in_include_idx = True
                break
            cur_node1 = morph.pos_dict[cur_node1][6]

        while cur_node2 != -1 and cur_node2 != -2:
            if cur_node2 in include_crossing_likely_idx:
                cur_node2_pair = True
                if not in_include_idx:
                    pair_num += 1
                pair_breakTip2_crossinglikely_idx.append(cur_node2)
                break
            cur_node2 = morph.pos_dict[cur_node2][6] 

        if cur_node1_pair and cur_node2_pair:
            break_pair_2cross_idx.append(cur_node1)
            break_pair_2cross_idx.append(cur_node2)

    pair_breakTip2_crossinglikely_idx = set(pair_breakTip2_crossinglikely_idx)
    break_pair_2cross_idx = set(break_pair_2cross_idx)
    print(f"There are {pair_num} break_tip2 pair crossinglikely, and breakTip2 match {len(pair_breakTip2_crossinglikely_idx)} crossinglikely !")
    return pair_breakTip2_crossinglikely_idx, break_pair_2cross_idx, pair_num

    

def find_breakTip2_corresCrossing(break_tip2_pairs, include_crossing_likely_idx, all_angles_joint_max, morph):
    # helper function
    def _find_tip_corres_maxCrossing(tip, crossing_likely_idx, score):
        ## 因为先排除了一部分类crossing，所以导致一些中断tip匹配不到类crossing，导致max_idx都是-2这样不好！！！！  还是应该把self-crossing这里提前
        max_idx = -2
        max_score = 0
        self_crossing_ii = -1
        while tip != morph.idx_soma:
            if tip == -1:
                break
            if tip in crossing_likely_idx:
                ii = np.argwhere(np.array(crossing_likely_idx)==tip)[0][0]
                if score[ii] > max_score:
                    max_idx = include_crossing_likely_idx[ii]
                    max_score = score[ii]
                    self_crossing_ii = ii
                
            tip = morph.pos_dict[tip][6]
        return max_idx, max_score, self_crossing_ii
            
    breakTip2_corresCrossing = []
    include_inter_crossing_likely_idx = copy.deepcopy(include_crossing_likely_idx)
    for tip1, tip2 in break_tip2_pairs:
        corres_idx1,corres_score1,self_crossing_ii1 = _find_tip_corres_maxCrossing(tip1, include_crossing_likely_idx, all_angles_joint_max)
        corres_idx2,corres_score2, self_crossing_ii2 = _find_tip_corres_maxCrossing(tip2, include_crossing_likely_idx, all_angles_joint_max)
        if corres_score1 > corres_score2:
            breakTip2_corresCrossing.append(((tip2,tip1), corres_idx1)) # 也就是tip1和类crossing连接，这样对点进行组合，方便后面的修正
            del include_inter_crossing_likely_idx[self_crossing_ii1] 
        else:
            breakTip2_corresCrossing.append(((tip1,tip2), corres_idx2)) # 也就是tip2和类crossing连接
            del include_inter_crossing_likely_idx[self_crossing_ii2]
    return breakTip2_corresCrossing, include_inter_crossing_likely_idx


def revise_inter_crossing(morph, include_inter_crossing_likely_idx, connections):
    ## 这里感觉只需要把另外两个不连接到三分叉点的子节点，断开就行了
    for inter_crossing_idx in include_inter_crossing_likely_idx:
        connection_point1,connection_point2 = connections[inter_crossing_idx]
        morph.pos_dict[connection_point1][6] = -2
        morph.pos_dict[connection_point2][6] = -2
    return morph

"""
def judging_connect_soma_notRecursion(idx, pos_dict, child_dict):
    idx, type_, x, y, z, r, p, connect_tosoma = pos_dict[idx]
    connect_tosoma = True
    pos_dict[idx] = (idx, type_, x, y, z, r, p, connect_tosoma)
    for cur_idx in child_dict[idx]:
        idx, type_, x, y, z, r, p, connect_tosoma = pos_dict[cur_idx]
        connect_tosoma = True
        pos_dict[cur_idx] = (idx, type_, x, y, z, r, p, connect_tosoma)
        idx = cur_idx
"""        
    

def prune_notConnenctToSoma(app2_swc_file, morph):
    ## 用DFS(或者BFS)比较好，毕竟这是天然的树结构
    def _judging_connect_soma(idx, pos_dict, child_dict):
        idx, type_, x, y, z, r, p, connect_tosoma = pos_dict[idx]
        connect_tosoma = True
        pos_dict[idx] = (idx, type_, x, y, z, r, p, connect_tosoma)

        if idx not in child_dict:
            ## DFS终止条件：到树的末端
            return
        
        for new_idx in child_dict[idx]:
            _judging_connect_soma(new_idx,pos_dict,child_dict)

    child_dict = {}
    pruned_pos_dict = copy.deepcopy(morph.pos_dict)
    pruned_pos_dict_temp = {}
    for idx, leaf in morph.pos_dict.items():
        idx, type_, x, y, z, r, p = leaf
        connect_tosoma = False
        leaf = (idx, type_, x, y, z, r, p, connect_tosoma)
        pruned_pos_dict_temp[idx] = leaf
        if leaf[-2] in child_dict:
            child_dict[leaf[-2]].append(leaf[0])
        else:
            child_dict[leaf[-2]] = [leaf[0]]
    ## 注意下面给_judging_connect_soma函数传递child_dict得时候不能用morph.child_dict，因为morph.child_dict是在一开始读进来Tree的时候的拓扑的child_dict，现在child_dict要用重新求的
    """
    if app2_swc_file == "/home/yjz/Project/Auto-tracing/crossing/Myself_newMethod/data/app2/maxR_rotate45_consensus/00000/7740_13735_4283_18453.swc":
        import ipdb; ipdb.set_trace()
    """

    try:
        _judging_connect_soma(morph.idx_soma, pruned_pos_dict_temp, child_dict)
    except RecursionError:
        print(f"RecursionError may be error!!! : {app2_swc_file}")
        #pass
        for idx, leaf in morph.pos_dict.items():
            idx, type_, x, y, z, r, p = leaf
            connect_tosoma = True
            leaf = (idx, type_, x, y, z, r, p, connect_tosoma)
            pruned_pos_dict_temp[idx] = leaf
            
    length = len(pruned_pos_dict)
    num = 0
    for key, value in pruned_pos_dict_temp.items():
        if value[-1] == False:
            num+=1
            pruned_pos_dict.pop(key)
    print(f"all_nodes {length}/ not {num}")
    return pruned_pos_dict
        
def pos_dict_toTree(pos_dict):
    tree = []
    for idx, type_, x, y, z, r, p in pos_dict.values():
        tree.append((idx, type_, x, y, z, r, p))
        
    return tree

def get_crossing_likely_dataset(app2_stps, app2_multifurcation_cor):
    if len(app2_stps) != 0 and len(app2_multifurcation_cor) != 0:
        ## cor就是把所有类crossing结构得坐标得到，距离soma近得，我们可以不处理
           
        crossing_likely_cor = app2_stps
        crossing_likely_cor = np.vstack((crossing_likely_cor,app2_multifurcation_cor))

        ## idx就是把所有类crossing结构得idx得到，可以用在后面取半径和判断topo层级使用
        crossing_likely_idx = app2_cur_tip_ids # 对于邻近二分叉来说：cur这个点就可以了，他更靠近soma，更有利于topo层级判断和取半径 
        crossing_likely_idx.extend(app2_multifurcation_idx)

        crossing_likely_pairs_andIdx = app2_pairs
        crossing_likely_pairs_andIdx.extend(app2_multifurcation_idx) 

        crossing_likely_radius = app2_cur_tip_radius
        crossing_likely_radius.extend(app2_multifurcation_radius)

    if len(app2_stps) != 0 and len(app2_multifurcation_cor) == 0:
        ## cor就是把所有类crossing结构得坐标得到，距离soma近得，我们可以不处理
            
        crossing_likely_cor = app2_stps

        ## idx就是把所有类crossing结构得idx得到，可以用在后面取半径和判断topo层级使用
        crossing_likely_idx = app2_cur_tip_ids # 对于邻近二分叉来说：cur这个点就可以了，他更靠近soma，更有利于topo层级判断和取半径 

        crossing_likely_pairs_andIdx = app2_pairs

        crossing_likely_radius = app2_cur_tip_radius

    if len(app2_stps) == 0 and len(app2_multifurcation_cor) != 0:
        ## cor就是把所有类crossing结构得坐标得到，距离soma近得，我们可以不处理
            
        crossing_likely_cor = app2_multifurcation_cor

        ## idx就是把所有类crossing结构得idx得到，可以用在后面取半径和判断topo层级使用
        crossing_likely_idx = app2_multifurcation_idx # 对于邻近二分叉来说：cur这个点就可以了，他更靠近soma，更有利于topo层级判断和取半径 

        crossing_likely_pairs_andIdx = app2_multifurcation_idx

        crossing_likely_radius = app2_multifurcation_radius

    return crossing_likely_cor, crossing_likely_idx, crossing_likely_pairs_andIdx, crossing_likely_radius

def revise_crossing_likely(morph, include_crossing_likely_idx, connections):
    for crossing_idx in include_crossing_likely_idx:
        if len(connections[crossing_idx]):
            #connection_point1,connection_point2, connection_point3 = connections[crossing_idx]
            connection_point1,connection_point2 = connections[crossing_idx]
            temp_connection_point1 = list(morph.pos_dict[connection_point1])
            temp_connection_point2 = list(morph.pos_dict[connection_point2])
            #temp_connection_point3 = list(morph.pos_dict[connection_point3])
            temp_connection_point1[6] = -2
            temp_connection_point2[6] = -2
            #temp_connection_point3[6] = -2
            morph.pos_dict[connection_point1] = tuple(temp_connection_point1)
            morph.pos_dict[connection_point2] = tuple(temp_connection_point2)
            #morph.pos_dict[connection_point3] = tuple(temp_connection_point3)

    return morph

def revise_crossing_likely_withRadiusAndIntensity(morph, include_crossing_likely_idx, connections):
    for crossing_idx in include_crossing_likely_idx:
        if len(connections[crossing_idx]):
            if len(connections[crossing_idx]) == 2:
                #connection_point1,connection_point2, connection_point3 = connections[crossing_idx]
                connection_point1,connection_point2 = connections[crossing_idx]
                temp_connection_point1 = list(morph.pos_dict[connection_point1])
                temp_connection_point2 = list(morph.pos_dict[connection_point2])
                #temp_connection_point3 = list(morph.pos_dict[connection_point3])
                temp_connection_point1[6] = -2
                temp_connection_point2[6] = -2
                #temp_connection_point3[6] = -2
                morph.pos_dict[connection_point1] = tuple(temp_connection_point1)
                morph.pos_dict[connection_point2] = tuple(temp_connection_point2)
                #morph.pos_dict[connection_point3] = tuple(temp_connection_point3)
            if len(connections[crossing_idx]) == 3:
                temp_connection_point = list(morph.pos_dict[nearest_s1])
                temp_connection_point[6] = -2
                morph.pos_dict[nearest_s1] = tuple(temp_connection_point)

                temp_connection_point = list(morph.pos_dict[nearest_s2])
                temp_connection_point[6] = -2
                morph.pos_dict[nearest_s2] = tuple(temp_connection_point)
        
                temp_connection_point = list(morph.pos_dict[nearest_s3])
                temp_connection_point[6] = -2
                morph.pos_dict[nearest_s3] = tuple(temp_connection_point)
                            
    return morph

def find_pNode_inBreakTip2(morph, break_tip2_pairs):
    def _judge_node_connect_toSoma(morph, tip):
        while tip != -1:
            if tip == morph.idx_soma:
                return True
            try:
                ## tip点可能会到前面设置的-2或者-3，也就是断开了
                tip = morph.pos_dict[tip][6]
            except KeyError:
                break
        return False

    break_tip2_pairs_connections = []
    pairs_soma = 0 ## 两个中断的tip点都连接到了soma上
    pairs_not_soma = 0 ## 两个中断的tip点都没有连接到soma上
    for tip1, tip2 in break_tip2_pairs:
        tip1_soma = _judge_node_connect_toSoma(morph, tip1)
        tip2_soma = _judge_node_connect_toSoma(morph, tip2)
        if tip1_soma and not tip2_soma:
            break_tip2_pairs_connections.append((tip1, tip2)) ## 元组的第一个元素是父节点
        elif tip2_soma and not tip1_soma:
            break_tip2_pairs_connections.append((tip2, tip1))
        elif tip1_soma and tip2_soma:
            pairs_soma += 1
        elif not tip1_soma and not tip2_soma:
            pairs_not_soma += 1
    return break_tip2_pairs_connections, pairs_soma, pairs_not_soma

def revise_self_crossing(morph, break_tip2_pairs_connections, connections):
    for tip1, tip2 in break_tip2_pairs_connections:
        #tip2 是子节点
        cur_node = tip2
        ## 翻转父子节点关系，就当成翻转链表就好了，morph.pos_dict[cur_node][-1]求出来的父节点其实就是链表中节点的next(cur_node->next)
        pre_node = -3 ## 这里为了避免和一直用的-2重复，改成用-3
        while cur_node != -2 and cur_node != -1 and cur_node != -3:
            ## 不到-2就还没有到了最后一个父节点了，根部
            next_node = morph.pos_dict[cur_node][6]
            temp = list(morph.pos_dict[cur_node])
            temp[6] = pre_node
            morph.pos_dict[cur_node] = tuple(temp)
            pre_node = cur_node
            cur_node = next_node

        temp = list(morph.pos_dict[tip2])
        temp[6] = tip1
        morph.pos_dict[tip2] = tuple(temp)

        ## 把类crossing区域的另外半段接过来
        if pre_node in connections:
            if len(connections[pre_node]):
                connection = connections[pre_node]
                if connection[0] == pre_node:
                    temp = list(morph.pos_dict[connection[1]])
                    temp[6] = connection[0]
                    morph.pos_dict[connection[1]] = temp
                else:
                    temp = list(morph.pos_dict[connection[0]])
                    temp[6] = connection[1]
                    morph.pos_dict[connection[0]] = temp
    return morph
            

def soma_detection(swc_file, soma_radius, soma_dist, soma_num):
    try:
        tree = parse_swc(swc_file)
    except ValueError:
        tree = parse_swc_upsample(swc_file) 

    if tree == []:
        return -1, 0

    morph = morphology.Morphology(tree, p_soma=-1)
    morph.get_critical_points()

    print(f'Number of tips: {len(morph.tips)} in file: {swc_file}')
    if len(morph.tips) > 1000:
        return -2, 0

    all_soma = {morph.idx_soma}
    for node in tree:
        near_soma = []
        cur_idx = node[0]
        if cur_idx == morph.idx_soma:
            continue
        cur_radius = node[5]
        if cur_radius >= soma_radius:
            for soma in all_soma:
                soma_cor = np.array(morph.pos_dict[soma][2:5])
                cur_cor = np.array(morph.pos_dict[cur_idx][2:5])
                cur_dist = np.linalg.norm(soma_cor - cur_cor)
                if cur_dist < soma_dist:
                    near_soma.append(True)
                    break
                else:
                    near_soma.append(False)
                 
            if True in near_soma:
                continue
            else:
                all_soma.add(cur_idx)

    return all_soma, morph
                

def cut(morph, all_soma):
    key_fiber = []
    key_fiber_len = []
    for idx in all_soma:
        if idx == morph.idx_soma:
            continue
        cur_idx = idx
        cur_length = 0
        while cur_idx != -1 and cur_idx != -2:
            key_fiber.append(cur_idx)
            if morph.pos_dict[cur_idx][6] != -1 and morph.pos_dict[cur_idx][6] != -2:
                cur_cor = np.array(morph.pos_dict[cur_idx][2:5])
                p_cor = np.array(morph.pos_dict[morph.pos_dict[cur_idx][6]][2:5])
                cur_length += np.linalg.norm(cur_cor - p_cor)
            key_fiber_len.append(cur_length)
            cur_idx = morph.pos_dict[cur_idx][6]     
        for i, length in enumerate(key_fiber_len):
            if length > cur_length / 2:
                cut_idx = key_fiber[i]
                break
        temp_point = list(morph.pos_dict[cut_idx]) 
        temp_point[6] = -2
        morph.pos_dict[cut_idx] = tuple(temp_point)

    return morph
        

if __name__ == '__main__':
    sys.setrecursionlimit(100000)

    maxR = True
    allData = True ## 只要不是那59个测试集，就都是True
    tiff_dir = "/media/yjz/4A0263A70263972B/SEU_Data/seu_mouse/crop_data/dendriteImageMaxR/tiff"
    #app2_swc_dir = "/home/yjz/Project/Auto-tracing/data/seu_mouse/crop_data/dendriteImageSecR_AllData/app2_upsample3"
    #post_swc_dir = "/home/yjz/Project/Auto-tracing/data/seu_mouse/crop_data/dendriteImageSecR_AllData/app2_upsample3_postprocessing"
    #app2_fused_swc_dir = "/home/yjz/Project/Auto-tracing/crossing/Myself_newMethod/data/app2/fused_tg0.0_alpha0.8_vanilla_bgMask0_upsample_to3pixel"
    #app2_swc_dir = "/home/yjz/Project/Auto-tracing/data/seu_mouse/crop_data/dendriteImageSecR_AllData/app2_upsample3"

    #app2_swc_dir = "/home/yjz/Project/Auto-tracing/data/seu_mouse/crop_data/dendriteImageMaxR_AllData/app2_upsample3"
    #post_swc_dir = "/home/yjz/Project/Auto-tracing/crossing/Myself_newMethod/data/post_processing/opt_changeParams_opt/opt_moreParams_optConnect_excludeMain/post_processing_optNeitherConnectToSoma_break_tip2/maxR_allData_OptJointMaxAngleThreshSection_thresh120"
    #app2_swc_dir = "/home/yjz/Project/Auto-tracing/crossing/Myself_newMethod/data/app2/maxR_rotate45_consensus"
    #app2_swc_dir = "/home/yjz/Project/Auto-tracing/crossing/Myself_newMethod/data/app2/fused_tg0.0_alpha0.8_vanilla_bgMask0_upsample_to3pixel"
    #post_swc_dir = "/home/yjz/Project/Auto-tracing/crossing/Myself_newMethod/data/post_processing/opt_changeParams_opt/opt_moreParams_optConnect_excludeMain/post_processing_optNeitherConnectToSoma_break_tip2/maxR_OptJointMaxAngleThreshSection_thresh120"
    app2_swc_dir = "/home/yjz/Project/Auto-tracing/data/seu_mouse/crop_data/dendriteImageMaxR_AllData/app2_upsample3_toyset_rotationAug"
    post_swc_dir = "/home/yjz/Project/Auto-tracing/crossing/Myself_newMethod/data/post_processing/add_moreFeatures/maxR_toyset_angleThresh10_withoutMoreFea_withoutClaw_addSomaDetCutLargerSomaNum5"

    all_dists = []

    app2_all_num = []
    #all_angles_joint_max = []
    app2_all_num_multif = []

    ## 也就是说下面得参数不分multif还是邻近二分叉，是统一得类crossing参数
    soma_threshold = 10.0
    ## 下面是一些数值统计变量
    all_pairs_soma = 0
    all_pairs_not_soma = 0
    all_break_tip2_pairs_num = 0
    all_crossing_likely_num = 0
    all_exclude_num = 0
    all_shortlySegment_num = 0
    ## 下面两个参数指的是，一个break只有一个对应的类crossing和一个break从两端分别对应了一个类crossing
    all_oneCross_pairBreak = 0
    all_twoCross_pairBreak = 0
    swc_num = 0
    all_pair_num = 0
    has_connection_num = 0
    ## 找中断点的参数
    dist_thresh = 1.0#1.5
    line_length = 5.0 # 这个是用在find_point_with_distance函数的
    ang_thresh = 140#120.0
    ## 比较短的分支
    shortlySegment_length = 6.0
    ## 找邻近二分叉的参数
    closing_bif_dist = 1.7#1.5
    ## exclude_main_branch
    #soma_threshold=12.0
    radius_threshold=5
    topo_threshshold=3
    ## 判断连接关系时大于多少度才满足我们的置信度
    joint_max_angle_thresh_2breakTip2 = 10#90
    joint_max_angle_thresh = 10#90 #常用120#140
    ##要不要在局部crossing区域增加radius和intensity特征
    local_judge_radius_andIntensity = False
    ## 如果三分叉是爪子形状的话，就直接跳过
    ## 暂时设置claw=True得时候 judge_radius_andIntensity必须是False
    judge_claw = False 
    claw_angle = 150
    claw_num = 0
    all_claw_num = 0
    ## 如果检查swc中超过1个soma，但是少于4个，则进行剪枝
    soma_detect = True
    ## 如果检查swc中超过1个soma，但是少于4个，则只保留中间的切掉其他的
    soma_detection_cut = True
    soma_cut_radius = 5
    soma_cut_dist = 150
    soma_cut_num = 10
    all_soma_count = 0
    one_soma_swc = 0
    small_soma_swc = 0
    large_soma_swc = 0
    ## 只有一个soma的时候是否进行prune
    one_soma = False
    ## 要不要从全局角度考虑增加radius和intensity特征 
    overall_judge_
    

    if maxR:
        soma_threshold *= 2
        dist_thresh *= 2
        line_length *= 2
        shortlySegment_length *= 2
        closing_bif_dist *= 2
        radius_threshold *= 2

    
    for brain in glob.glob(os.path.join(app2_swc_dir,"[1-9]*[1-9]")):
        brain_id = os.path.split(brain)[-1]
        for app2_swc_file in glob.glob(os.path.join(brain,'*.swc')):
            if allData:
                if not os.path.exists(os.path.join(post_swc_dir,brain_id)):
                    os.mkdir(os.path.join(post_swc_dir,brain_id))
                swc_name = os.path.split(app2_swc_file)[-1]
                swc_prefix = os.path.splitext(swc_name)[0]
                swc_name = swc_prefix + "_" + brain_id + ".swc" 
            else:
                swc_name = os.path.split(app2_swc_file)[-1]
                brain_id = os.path.splitext(swc_name)[0].split("_")[-1]
                swc_brain_prefix = os.path.splitext(swc_name)[0]
                swc_brain_prefix_split = swc_brain_prefix.split("_")
                swc_prefix = swc_brain_prefix_split[0] + "_" + swc_brain_prefix_split[1] + "_" + swc_brain_prefix_split[2]
            #app2_swc_file = os.path.join(app2_swc_dir,brain_id,swc_prefix+".swc")
    
            #if swc_prefix == "21357_12301_5813":
            #    import ipdb; ipdb.set_trace()
            
            if soma_detect:
                all_soma, morph = soma_detection(app2_swc_file, soma_cut_radius, soma_cut_dist, soma_cut_num) 
                if all_soma == -1:
                    print(f"{swc_name} if a blank swc file!")
                    continue
                if all_soma == -2:
                    print(f"There are massive tips in {swc_name} file!")
                    continue
            
                print(f"There are {len(all_soma)} somas in {swc_name}!")
                all_soma_count += len(all_soma)

                if len(all_soma) > soma_cut_num:
                    large_soma_swc += 1
                    print(f"There are massive somas in {swc_name} file!") 
                    continue
                if one_soma:
                    one_soma_swc += 1
                    if len(all_soma) == 1:
                        print(f"There is only one soma in {swc_name} file!")
                        continue

                small_soma_swc += 1 

            if soma_detection_cut and len(all_soma) > 1:
                print("Soma Detection Cut:")
                cut_morph = cut(morph, all_soma)                 
                pruned_pos_dict = prune_notConnenctToSoma(app2_swc_file, cut_morph)
                #cut_morph.pos_dict = pruned_pos_dict 
                pruned_tree = pos_dict_toTree(pruned_pos_dict)
                cut_morph = morphology.Morphology(pruned_tree, p_soma=-1)
                cut_morph.get_critical_points()
                
            print("Post Pruned:")
            claw_num = 0
            
            connections = {}
            all_angles_joint_max = []
            crossing_likely_num = []
            ## 没有带multif或者multifurcation的就是邻近二分叉
            if soma_detection_cut and len(all_soma) > 1:
                app2_stps, app2_pairs,app2_cur_tip_ids, app2_cur_tip_radius, app2_morph,app2_num = calc_dist_furc_addSomaDetCut(cut_morph, app2_swc_file, soma_threshold, closing_bif_dist)
            else:
                app2_stps, app2_pairs,app2_cur_tip_ids, app2_cur_tip_radius, app2_morph,app2_num = calc_dist_furc(app2_swc_file, soma_threshold, closing_bif_dist)

            ## =-1 意思是这个文件是空的
            if app2_num == -1:
                print(f"{swc_name} if a blank swc file!")
                continue
            ## 末端点过多，不合常理
            if app2_num == -2:
                print(f"There are massive tips in {swc_name} file!")
                continue

            #swc_num += 1            

            app2_multifurcation_cor, app2_multifurcation_idx, app2_multifurcation_radius, app2_multifurcation_num = find_multifucation(app2_morph)

            print(f"There are {app2_num} closing_bifurcation, and {app2_multifurcation_num} multifurcation")

            app2_all_num.append(app2_num)
            
            app2_all_num_multif.append(app2_multifurcation_num)

            crossing_likely_num.append(app2_num)
            crossing_likely_num.append(app2_multifurcation_num)

            if len(app2_stps) == 0 and len(app2_multifurcation_cor) == 0:
                print(f"There are not exist crossing_likely in {swc_name} file!")
                continue

            swc_num += 1            

            crossing_likely_cor, crossing_likely_idx, crossing_likely_pairs_andIdx, crossing_likely_radius = get_crossing_likely_dataset(app2_stps, app2_multifurcation_cor)


            ##1 首先将一部分符合条件的类crossing拿出去，不需要处理
            include_crossing_likely_idx,include_crossing_likely_pairs_idx, include_crossing_likely_cor, exclude_num = exclude_main_branch(crossing_likely_cor, crossing_likely_idx, crossing_likely_pairs_andIdx, crossing_likely_radius, app2_morph, soma_threshold, radius_threshold, topo_threshshold)
            
            all_crossing_likely_num += len(crossing_likely_idx)
            all_exclude_num += exclude_num
            print(f"There are {len(crossing_likely_idx)} crossing_likely, and {exclude_num} do not need to process!")
            ##2 删掉分支很短的(可能是bouton) 其实这个地方可以针对所有的二分叉，而不是仅仅类crossing????，就是不知道追踪出来的结果，这个到底多不多。这里可以改一下，先不用全部的二分叉三分叉，用所有的类crossing，也不是用所有类crossing的子集：include_crossing_likely
            post_morph, shortlySegment_num = delete_shortlySegment(app2_morph, crossing_likely_pairs_andIdx, length=shortlySegment_length)  
            all_shortlySegment_num += shortlySegment_num

            ##3 找所有中断点
            ## 这里主要参考post_crossing.py中main函数和generate_tip_error_ano_file()的写法
            #import ipdb; ipdb.set_trace()
            if soma_detection_cut and len(all_soma) > 1:
                break_tip2_pairs = find_breakTip2Pairs_somaDetCut(app2_swc_file, post_morph, dist_thresh, line_length, ang_thresh)
            else:
                break_tip2_pairs = find_breakTip2Pairs(app2_swc_file, dist_thresh, line_length, ang_thresh)
            all_break_tip2_pairs_num += len(break_tip2_pairs)

            ##3.5 感觉这里可以加一个函数：break_tip2和类crossing配对的，哪些可以组成对，这样可以组成对的类crossing必须给出连接关系，无法组成对的挑置信度大的连接关系(但是还有个问题就是可能大量的类crossing都对应到了break_tip2可以看效果)
            ## 然后下面给连接关系的时候，只有在pair_breakTip2_crossinglikely_idx中的和置信度大的，才给连接关系
            pair_breakTip2_crossinglikely_idx, break_pair_2cross_idx, pair_num = pair_breakTip2_crossinglikely(post_morph, break_tip2_pairs, include_crossing_likely_idx)
            pair_breakTip2_crossinglikely_idx -= break_pair_2cross_idx
            all_pair_num += pair_num
            all_twoCross_pairBreak += len(break_pair_2cross_idx)
            all_oneCross_pairBreak += len(pair_breakTip2_crossinglikely_idx) / 2

            ##4 所有crossing求score(暂时以angle作为得分)，并且此时就应该把连接关系给出来,要给出max_joint角度(用来判断中断tip和路径上哪个crossing匹配)，join_max对应的点的连接，因为修正中断的tip对应的crossing(self-crossing)的时候，要把拓扑关系改过来，而且没有匹配上的crossing如果不是直接删除的话也要这个拓扑关系(来保证其中一条不被删除)
            for i, idx in enumerate(include_crossing_likely_pairs_idx):
                if type(idx) == tuple:
                    astp_or_multif_cor = include_crossing_likely_cor[i]
                    ## point_X是根据dist找的点，用于后面求angle的，nearest_X最靠近关键点(pre_tip_id\cur_tip_id\multifurcation)的父节点或者子节点
                    #(point_p,nearest_p), (point_s1,nearest_s1),(point_s2,nearest_s2),(point_s3,nearest_s3) = find_points_forangles(astp_or_multif_cor,idx[0],idx[1],app2_morph) 
                    (point_p,nearest_p), (point_s1,nearest_s1),(point_s2,nearest_s2),(point_s3,nearest_s3) = find_points_forangles(astp_or_multif_cor,idx[0],idx[1],post_morph) 
                    #
                    ## 加两个判断条件： 从父->子的radius和intensity是递减的，所以感觉这里可以给nearest_X加一个状态变量，如果不满足递减条件，那么状态变量就是false，判断连接关系的时候，父段一定不和状态是false的子段相连接
                    decreasing_s1 = True
                    decreasing_s2 = True
                    decreasing_s3 = True
                    #local_judge_radius_andIntensity = False
                    #import ipdb; ipdb.set_trace()
                    if local_judge_radius_andIntensity:
                        tiff_file = os.path.join(tiff_dir, brain_id, swc_prefix+".tiff")
                        decreasing_s1, decreasing_s2, decreasing_s3 = local_judge_radius_andIntensity_decreasing(post_morph, tiff_file, nearest_p, nearest_s1, nearest_s2, nearest_s3)
                    if local_judge_radius_andIntensity:
                        if decreasing_s1 or decreasing_s2 or decreasing_s3:
                            angles_joint_max, joint_max_angle2, connection, connection_temp = calc_angles_withRadiusAndIntensity(astp_or_multif_cor,point_p,nearest_p, point_s1,nearest_s1,point_s2,nearest_s2,point_s3,nearest_s3, decreasing_s1, decreasing_s2, decreasing_s3)
                        else:
                            ## 子段的radius或intensity全都比父段的大，直接全部断开就好(应该是inter的)
                            connection = (nearest_s1,nearest_s2, nearest_s3)
                            """
                            temp_connection_point = list(app2_morph.pos_dict[nearest_s1])
                            temp_connection_point[6] = -2
                            app2_morph.pos_dict[nearest_s1] = tuple(temp_connection_point)

                            temp_connection_point = list(app2_morph.pos_dict[nearest_s2])
                            temp_connection_point[6] = -2
                            app2_morph.pos_dict[nearest_s2] = tuple(temp_connection_point)
        
                            temp_connection_point = list(app2_morph.pos_dict[nearest_s3])
                            temp_connection_point[6] = -2
                            app2_morph.pos_dict[nearest_s3] = tuple(temp_connection_point)
                            """
                        
                    else:
                        if judge_claw:
                            angles_joint_max, joint_max_angle2, connection, connection_temp, claw = calc_angles_withClaw(astp_or_multif_cor,point_p,nearest_p, point_s1,nearest_s1,point_s2,nearest_s2,point_s3,nearest_s3, claw_angle)
                            if claw:
                                claw_num += 1
                                all_claw_num += 1
                        else:
                            angles_joint_max, joint_max_angle2, connection, connection_temp = calc_angles(astp_or_multif_cor,point_p,nearest_p, point_s1,nearest_s1,point_s2,nearest_s2,point_s3,nearest_s3)
                    #connections.update({connection[0]:connection})
                    #connections.update({connection[1]:connection})
                    #connections.update({idx[0]:connection_temp})
                    #connections.update({idx[1]:connection_temp})
                    #if idx[0] in pair_breakTip2_crossinglikely_idx or idx[1] in pair_breakTip2_crossinglikely_idx or (joint_max_angle2[0] >= joint_max_angle_thresh and joint_max_angle2[1] >= joint_max_angle_thresh):
                    if local_judge_radius_andIntensity:
                        if decreasing_s1 or decreasing_s2 or decreasing_s3:
                            if ((idx[0] in break_pair_2cross_idx or idx[1] in break_pair_2cross_idx) and (joint_max_angle2[0] >= joint_max_angle_thresh_2breakTip2 and joint_max_angle2[1] >= joint_max_angle_thresh_2breakTip2)) or ((idx[0] in pair_breakTip2_crossinglikely_idx or idx[1] in pair_breakTip2_crossinglikely_idx) and (joint_max_angle2[0] >= joint_max_angle_thresh and joint_max_angle2[1] >= joint_max_angle_thresh)) or (joint_max_angle2[0] >= joint_max_angle_thresh_2breakTip2 and joint_max_angle2[1] >= joint_max_angle_thresh_2breakTip2):
                                connections.update({idx[0]:connection})
                                connections.update({idx[1]:connection})
                                has_connection_num += 1
                            else:
                                connection = tuple()
                                connections.update({idx[0]:connection})
                                connections.update({idx[1]:connection})
                        else:
                            #connection = tuple()
                            connections.update({idx[0]:connection})
                            connections.update({idx[1]:connection})
                            has_connection_num += 1
                    else:
                        if judge_claw:
                            if claw:
                                connection = tuple()
                                connections.update({idx[0]:connection})
                                connections.update({idx[1]:connection})
                            else:
                                if ((idx[0] in break_pair_2cross_idx or idx[1] in break_pair_2cross_idx) and (joint_max_angle2[0] >= joint_max_angle_thresh_2breakTip2 and joint_max_angle2[1] >= joint_max_angle_thresh_2breakTip2)) or ((idx[0] in pair_breakTip2_crossinglikely_idx or idx[1] in pair_breakTip2_crossinglikely_idx) and (joint_max_angle2[0] >= joint_max_angle_thresh and joint_max_angle2[1] >= joint_max_angle_thresh)) or (joint_max_angle2[0] >= joint_max_angle_thresh_2breakTip2 and joint_max_angle2[1] >= joint_max_angle_thresh_2breakTip2):
                        ## 其实这个判断好像弄麻烦了，就是两个：在pair_breakTip2_crossinglikely_idx中要起码大于120度，不在这里的(包括了break_tip2从两端都对应了类crossing的情况)必须都大于150就好了
                                    connections.update({idx[0]:connection})
                                    connections.update({idx[1]:connection})
                                    has_connection_num += 1
                                else:
                                    connection = tuple()
                                    connections.update({idx[0]:connection})
                                    connections.update({idx[1]:connection})
                                
                        else:
                            if ((idx[0] in break_pair_2cross_idx or idx[1] in break_pair_2cross_idx) and (joint_max_angle2[0] >= joint_max_angle_thresh_2breakTip2 and joint_max_angle2[1] >= joint_max_angle_thresh_2breakTip2)) or ((idx[0] in pair_breakTip2_crossinglikely_idx or idx[1] in pair_breakTip2_crossinglikely_idx) and (joint_max_angle2[0] >= joint_max_angle_thresh and joint_max_angle2[1] >= joint_max_angle_thresh)) or (joint_max_angle2[0] >= joint_max_angle_thresh_2breakTip2 and joint_max_angle2[1] >= joint_max_angle_thresh_2breakTip2):
                        ## 其实这个判断好像弄麻烦了，就是两个：在pair_breakTip2_crossinglikely_idx中要起码大于120度，不在这里的(包括了break_tip2从两端都对应了类crossing的情况)必须都大于150就好了
                                connections.update({idx[0]:connection})
                                connections.update({idx[1]:connection})
                                has_connection_num += 1
                            else:
                                connection = tuple()
                                connections.update({idx[0]:connection})
                                connections.update({idx[1]:connection})
                
                else:
                    astp_or_multif_cor = include_crossing_likely_cor[i]
                    #(point_p,nearest_p), (point_s1,nearest_s1),(point_s2,nearest_s2),(point_s3,nearest_s3) = find_points_forangles_multif(astp_or_multif_cor, idx, app2_morph) 
                    (point_p,nearest_p), (point_s1,nearest_s1),(point_s2,nearest_s2),(point_s3,nearest_s3) = find_points_forangles_multif(astp_or_multif_cor, idx, post_morph) 
                    #
                    ## 加两个判断条件： 从父->子的radius和intensity是递减的，所以感觉这里可以给nearest_X加一个状态变量，如果不满足递减条件，那么状态变量就是false，判断连接关系的时候，父段一定不和状态是false的子段相连接
                    decreasing_s1 = True
                    decreasing_s2 = True
                    decreasing_s3 = True
                    #local_judge_radius_andIntensity = True
                    if local_judge_radius_andIntensity:
                        tiff_file = os.path.join(tiff_dir, brain_id, swc_prefix+".tiff")
                        decreasing_s1, decreasing_s2, decreasing_s3 = local_judge_radius_andIntensity_decreasing(post_morph, tiff_file, nearest_p, nearest_s1, nearest_s2, nearest_s3)
            
                    if local_judge_radius_andIntensity:
                        if decreasing_s1 or decreasing_s2 or decreasing_s3:
                            angles_joint_max, joint_max_angle2, connection, connection_temp = calc_angles_withRadiusAndIntensity(astp_or_multif_cor,point_p,nearest_p, point_s1,nearest_s1,point_s2,nearest_s2,point_s3,nearest_s3, decreasing_s1, decreasing_s2, decreasing_s3)
                        else:
                            connection = {nearest_s1, nearest_s2, nearest_s3}
                            ## 子段的radius或intensity全都比父段的大，直接全部断开就好(应该是inter的)
                            """
                            temp_connection_point = list(app2_morph.pos_dict[nearest_s1])
                            temp_connection_point[6] = -2
                            app2_morph.pos_dict[nearest_s1] = tuple(temp_connection_point)

                            temp_connection_point = list(app2_morph.pos_dict[nearest_s2])
                            temp_connection_point[6] = -2
                            app2_morph.pos_dict[nearest_s2] = tuple(temp_connection_point)
        
                            temp_connection_point = list(app2_morph.pos_dict[nearest_s3])
                            temp_connection_point[6] = -2
                            app2_morph.pos_dict[nearest_s3] = tuple(temp_connection_point)
                            """
                            
                    else:
                        if judge_claw:
                            angles_joint_max, joint_max_angle2, connection, connection_temp, claw = calc_angles_withClaw(astp_or_multif_cor,point_p,nearest_p, point_s1,nearest_s1,point_s2,nearest_s2,point_s3,nearest_s3, claw_angle)
                            if claw:
                                claw_num += 1
                                all_claw_num += 1
                        else:
                            angles_joint_max, joint_max_angle2, connection, connection_temp = calc_angles(astp_or_multif_cor,point_p,nearest_p, point_s1,nearest_s1,point_s2,nearest_s2,point_s3,nearest_s3)
                    #connections.update({connection[0]:connection})
                    #connections.update({connection[1]:connection})
                    #connections.update({idx:connection_temp})
                    #if idx in pair_breakTip2_crossinglikely_idx or (joint_max_angle2[0] >= joint_max_angle_thresh and joint_max_angle2[1] >= joint_max_angle_thresh):
                    if local_judge_radius_andIntensity:
                        if decreasing_s1 or decreasing_s2 or decreasing_s3:
                            if (idx in pair_breakTip2_crossinglikely_idx and (joint_max_angle2[0] >= joint_max_angle_thresh and joint_max_angle2[1] >= joint_max_angle_thresh)) or (joint_max_angle2[0] >= joint_max_angle_thresh_2breakTip2 and joint_max_angle2[1] >= joint_max_angle_thresh_2breakTip2):
                                connections.update({idx:connection})
                                has_connection_num += 1
                            else:
                                connection = tuple()
                                connections.update({idx:connection})
                        else:
                            #connection = tuple()
                            connections.update({idx:connection})
                            has_connection_num += 1
                        
                    else:
                        if judge_claw:
                            if claw:
                                connection = tuple()
                                connections.update({idx:connection})
                            else:
                                if (idx in pair_breakTip2_crossinglikely_idx and (joint_max_angle2[0] >= joint_max_angle_thresh and joint_max_angle2[1] >= joint_max_angle_thresh)) or (joint_max_angle2[0] >= joint_max_angle_thresh_2breakTip2 and joint_max_angle2[1] >= joint_max_angle_thresh_2breakTip2):
                              
                                    ## 其实这个判断好像弄麻烦了，就是两个：在pair_breakTip2_crossinglikely_idx中要起码大于120度，不在这里的(包括了break_tip2从两端都对应了类crossing的情况)必须都大于150就好了
                                    connections.update({idx:connection})
                                    has_connection_num += 1
                                else:
                                    connection = tuple()
                                    connections.update({idx:connection})
                        else:
                            if (idx in pair_breakTip2_crossinglikely_idx and (joint_max_angle2[0] >= joint_max_angle_thresh and joint_max_angle2[1] >= joint_max_angle_thresh)) or (joint_max_angle2[0] >= joint_max_angle_thresh_2breakTip2 and joint_max_angle2[1] >= joint_max_angle_thresh_2breakTip2):
                                connections.update({idx:connection})
                                has_connection_num += 1
                            else:
                                connection = tuple()
                                connections.update({idx:connection})
                
                ## 利用list特性，传进去的参数会随着函数的作用一起改变
                #get_all_angles(all_angles_joint_max, angles_joint_max)
                
            if judge_claw:
                print(f"There are {claw_num} claws in this swc!")

            ##5 对所有的类crossing进行矫正，这里其实和当初的revise_inter_crossing作用一样：只是当初只针对inter-crossing，但是现在不管self-crossing还是inter-crossing，都先进行修正，但是现在因为还没有跟后面中断的tip对进行匹配，所以统一把另外两个没有连接到父段的子段断开即可，后面在中断点那如果这些断开的子段和中断点相连了，那他就会被连起来而且接到了soma上，其余断掉的子段没有跟中断点匹配所以没有连接到soma上，会被当作inter-crossing然后被剪枝掉。
            #import ipdb; ipdb.set_trace()
            ##  这里只把有对应break_tip2的和置信度大的类crossing进行revise
            if local_judge_radius_andIntensity:
                post_morph = revise_crossing_likely_withRadiusAndIntensity(post_morph, include_crossing_likely_idx, connections)
            else:
                post_morph = revise_crossing_likely(post_morph, include_crossing_likely_idx, connections)

            ##6 将两个中断点重连并且如果有第4步中断掉的子段匹配，那就一起连上，并且父子关系该翻转就翻转。
            #首先这个函数用来判断两个中断点，到底谁是父节点(因为前面对类crossing结构进行了结构修正，所以理论上有一些中断点就不会再接到soma上了)            
            break_tip2_pairs_connections, pairs_soma, pairs_not_soma = find_pNode_inBreakTip2(post_morph, break_tip2_pairs)   
            all_pairs_soma += pairs_soma
            all_pairs_not_soma += pairs_not_soma
            print(f"There are {len(break_tip2_pairs)} break_tip2_pairs, {pairs_soma} break_tip2_pairs all connect to soma(lead to ring), and {pairs_not_soma} break_tip2_pairs neither connect to soma!")
            #这个函数对中断点这里进行修正(也就是把对应子节点的tip接到父节点tip上，并且子节点后面理论上还有在类crossing那里被断掉的子段，也需要拼接起来并且肚子关系需要翻转)
            post_morph = revise_self_crossing(post_morph, break_tip2_pairs_connections, connections) 

            ##7 所有没有连接到soma的点进行prune(前面self-crossing的已经和中断的tip点连接到一起了，剩下没有连接到soma的就是Inter-crossing了)
            pruned_pos_dict = prune_notConnenctToSoma(app2_swc_file, post_morph)
            ##8 sort_swc
            pruned_tree = pos_dict_toTree(pruned_pos_dict)
            if allData:
                write_swc(pruned_tree, os.path.join(post_swc_dir,brain_id,swc_prefix+"_revised.swc"))
            else:
                write_swc(pruned_tree, os.path.join(post_swc_dir,swc_prefix+"_revised.swc"))

            print("Finished!") 
            print("\n")

    print(f"There are {swc_num} swc files!")
    print(f"There are {all_crossing_likely_num} crossing_likely, and {all_exclude_num} do not need to revise!")
    print(f"There are {has_connection_num} crossing_likely have connections!")
    print(f"There are {all_pair_num} pair_breakTip2_crossinglikely!")
    print(f"There are {all_oneCross_pairBreak} oneCross_pairBreak, and {all_twoCross_pairBreak} twoCross_pairBreak !")
    print(f"There are {np.sum(app2_all_num)} closing_bifurcation, and {np.sum(app2_all_num_multif)} multifurcation!")
    print(f"There are {all_crossing_likely_num} crossing_likely, and {all_shortlySegment_num} have short segment!")
    print(f"There are {all_break_tip2_pairs_num} break_tip2_pairs, {all_pairs_soma} pairs all connect to soma(lead to ring), and {all_pairs_not_soma} break_tip2_pairs neither connect to soma!") 
    if soma_detect:
        print(f"There are {all_soma_count} somas, avg: {all_soma_count / swc_num}!")
        print(f"There are {one_soma_swc} swc files have one soma; {small_soma_swc} swc files have smaller than thresh somas; {large_soma_swc} swc files have larger than thresh somas")
    if judge_claw:
        print(f"There are {all_claw_num} claws!")
    print("All finished!")
            
            
