import os, sys
import glob
import copy
import numpy as np


from swc_handler import parse_swc, parse_swc_upsample, write_swc, find_soma_node
import morphology


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
        

if __name__ == "__main__":
    app2_swc_dir = "/home/yjz/Project/Auto-tracing/data/seu_mouse/crop_data/dendriteImageMaxR_AllData/app2_upsample3_toyset"
    post_swc_dir = "/home/yjz/Project/Auto-tracing/crossing/Myself_newMethod/data/post_processing/soma_detection_cut/maxR_toyset_largerSomaNum8"

    soma_radius = 10
    soma_dist = 150
    soma_num = 8

    n = 0

    all_soma_count = 0
    one_soma_swc = 0
    small_soma_swc = 0
    large_soma_swc = 0
        
    
    for brain in glob.glob(os.path.join(app2_swc_dir, "*")):
        brain_id = os.path.split(brain)[-1]
        #if brain_id != "18454":
        #    continue
        for app2_swc_file in glob.glob(os.path.join(brain,"*swc")):
            n += 1;
            swc_name = os.path.split(app2_swc_file)[-1]
            swc_prefix = os.path.splitext(swc_name)[0]
            print(f"[{n}] Processing {brain_id}_{swc_name}:")
            #if swc_prefix != "21357_12301_5813":
            #    continue
            
            #import ipdb; ipdb.set_trace()
            all_soma, morph = soma_detection(app2_swc_file, soma_radius, soma_dist, soma_num)
            if all_soma == -1:
                print(f"{swc_name} if a blank swc file!")
                continue
            if all_soma == -2:
                print(f"There are massive tips in {swc_name} file!")
                continue
            
            print(f"There are {len(all_soma)} somas in {swc_name}!")
            all_soma_count += len(all_soma)

            if len(all_soma) > soma_num:
                large_soma_swc += 1
                print(f"There are massive somas in {swc_name} file!") 
                continue
            if len(all_soma) == 1:
                one_soma_swc += 1
                print(f"There is only one soma in {swc_name} file!")
                continue

            small_soma_swc += 1
            
            if not os.path.exists(os.path.join(post_swc_dir, brain_id)):
                os.mkdir(os.path.join(post_swc_dir, brain_id))

            cut_morph = cut(morph, all_soma)
            pruned_pos_dict = prune_notConnenctToSoma(app2_swc_file, cut_morph)
            pruned_tree = pos_dict_toTree(pruned_pos_dict)
            write_swc(pruned_tree, os.path.join(post_swc_dir,brain_id,swc_prefix+"_revised.swc")) 
            print("Finished!")
            print("\n")

    print(f"There are {n} swc files, and have {all_soma_count} somas, avg: {all_soma_count / n}!")
    print(f"There are {one_soma_swc} swc files have one soma; {small_soma_swc} swc files have smaller than thresh somas; {large_soma_swc} swc files have larger than thresh somas")
            

