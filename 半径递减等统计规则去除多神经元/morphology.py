#!/usr/bin/env python

#================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : morphology.py
#   Author       : Yufeng Liu
#   Date         : 2021-07-16
#   Description  : The script is responsible for morphological and 
#                  topological feature calculation, including: 
#                   - morphological: 
#                   - topological:
#                  The features: #stems, #tips, #total_length, #depth and #branches
#                  are the same as that in Vaa3D, tested on two randomly selected
#                  swc file. The #bifurcation of our package does not include 
#                  the soma point, while Vaa3D does.
#
#================================================================

import copy
import numpy as np
import sys
from swc_handler import get_child_dict, get_index_dict, find_soma_node, find_soma_index, NEURITE_TYPES

sys.setrecursionlimit(100000)

class AbstractTree(object):
    def __init__(self, tree, p_soma=-1):
        self.p_soma = p_soma

        self.tree = tree    # swc tree file, type as list
        self.child_dict = get_child_dict(tree)
        self.index_dict = get_index_dict(tree)
        self.pos_dict = self.get_pos_dict()
        self.idx_soma = find_soma_node(tree, p_soma=self.p_soma)    # node index
        self.index_soma = find_soma_index(tree, p_soma)


    def get_nodes_by_types(self, neurite_type):
        nodes = []
        ntypes = NEURITE_TYPES[neurite_type]
        for node in self.tree:
            type_ = node[1]
            if type_ in ntypes:
                nodes.append(node[0])
        return set(nodes)

    def get_pos_dict(self):
        pos_dict = {}
        for i, leaf in enumerate(self.tree):
            pos_dict[leaf[0]] = leaf
        return pos_dict

    def get_volume_size(self):
        coords = [leaf[2:5] for leaf in self.tree]
        coords = np.array(coords)
        cmin = coords.min(axis=0)
        cmax = coords.max(axis=0)
        span = cmax - cmin
        volume = span.prod()
        return span, volume

    def calc_node_distances(self):
        """
        Distance distribution for connecting nodes
        """
        coords1 = []
        coords2 = []
        for idx in self.child_dict:
            if idx == self.p_soma: continue
            coord1 = self.pos_dict[idx][2:5]
            for c_idx in self.child_dict[idx]:
                coord2 = self.pos_dict[c_idx][2:5]
                coords1.append(coord1)
                coords2.append(coord2)
        coords1 = np.array(coords1)
        coords2 = np.array(coords2)
        shift = coords2 - coords1
        dists = np.linalg.norm(shift, axis=1)
        print(dists.shape)
        stats = dists.mean(), dists.std(), dists.max(), dists.min()
        return stats

    def get_distances_to_soma(self):
        """
        distance to soma for all nodes
        """
        c_soma = np.array(self.pos_dict[self.idx_soma][2:5])
        coords = np.array([node[2:5] for node in self.tree])
        diff = coords - c_soma
        dists = np.linalg.norm(diff, axis=1)
        return dists

    def get_critical_points(self):
        # stems
        self.stems = set(self.child_dict[self.idx_soma])

        # terminal points
        all_nodes_indices = set([leaf[0] for leaf in self.tree])
        has_child_indices = set(list(self.child_dict.keys()))
        self.tips = all_nodes_indices - has_child_indices

        # and bifurcate points
        self.unifurcation = []   # with only one child
        self.bifurcation = []   # 2 childs
        self.multifurcation = [] # > 2 childs
        for idx, childs in self.child_dict.items():
            if idx == self.p_soma:
                continue    # virtual point for soma parent
            if idx == self.idx_soma:
                continue
            if len(childs) == 1:
                self.unifurcation.append(idx)
            elif len(childs) == 2:
                self.bifurcation.append(idx)
            elif len(childs) > 2:
                self.multifurcation.append(idx)
        self.unifurcation = set(self.unifurcation)
        self.bifurcation = set(self.bifurcation)
        self.multifurcation = set(self.multifurcation)

    def get_all_paths(self):
        """
        Find out all paths from tip to soma
        """
        if not hasattr(self, 'tips'):
            self.get_critical_points()

        paths = {}
        for tip in self.tips:
            path = [tip]
            leaf = self.pos_dict[tip]
            while leaf[-1] in self.pos_dict:
                pid = leaf[-1]
                path.append(pid)
                leaf = self.pos_dict[pid]
            paths[tip] = path
        return paths
        

    def calc_seg_lengths(self):
        # in parallel mode
        coords = np.array([leaf[2:5] for leaf in self.tree])
        p_coords = np.array([self.pos_dict[leaf[-1]][2:5] if leaf[0] != self.idx_soma else self.pos_dict[self.idx_soma][2:5] for leaf in self.tree])
        vectors = coords - p_coords
        lengths = np.linalg.norm(vectors, axis=1)
        return lengths

    def calc_total_length(self):
        seg_lengths = self.calc_seg_lengths()
        total_length = seg_lengths.sum()
        return total_length


class Morphology(AbstractTree):
    def __init__(self, tree, p_soma=-1):
        super(Morphology, self).__init__(tree, p_soma=p_soma)

    def get_path_idx_dict(self):
        def find_path_dfs(idx, path_dict, pos_dict, child_dict):
            pidx = pos_dict[idx][-1]
            
            if pidx in path_dict:
                path_dict[idx] = path_dict[pidx] + [pidx]

            if idx not in child_dict:
                return
            else:
                for cidx in child_dict[idx]:
                    find_path_dfs(cidx, path_dict, pos_dict, child_dict)

        path_dict = {}
        path_dict[self.idx_soma] = []
        find_path_dfs(self.idx_soma, path_dict, self.pos_dict, self.child_dict)
        return path_dict

    def get_path_len_dict(self, path_dict, seg_lengths):
        plen_dict = {}
        for idx, pidxs in path_dict.items():
            pindex = [self.index_dict[pidx] for pidx in pidxs]
            plen = seg_lengths[pindex].sum()
            plen_dict[idx] = plen
        return plen_dict

    def convert_to_topology_tree(self):
        """
        The original tree contains unifurcation, which should be merged
        """
        def convert_dfs(idx, child_dict, unifurcation, pos_dict_copy, new_tree):
            leaf = pos_dict_copy[idx]
            if idx not in child_dict:
                new_tree.append(leaf)
                return 

            if idx in unifurcation:
                # delete current node
                # change all childs's parent node inplace
                p_idx = leaf[-1]
                for child_idx in child_dict[idx]:
                    tmp_leaf = pos_dict_copy[child_idx]
                    tmp_leaf = (*tmp_leaf[:-1], p_idx)
                    pos_dict_copy[child_idx] = tmp_leaf
            else:
                new_tree.append(leaf)

            for child_idx in child_dict[idx]:
                convert_dfs(child_idx, child_dict, unifurcation, pos_dict_copy, new_tree)
                    


        new_tree = []
        pos_dict_copy = copy.deepcopy(self.pos_dict)
        convert_dfs(self.idx_soma, self.child_dict, self.unifurcation, pos_dict_copy, new_tree)
        print(f'{len(new_tree)} #nodes left after merging of the original {len(self.tree)} # nodes')
        return new_tree


class Topology(AbstractTree):
    def __init__(self, tree, p_soma=-1):
        super(Topology, self).__init__(tree, p_soma=p_soma)
        self.get_critical_points()
        self.calc_order_dict()

    def calc_order_dict(self):
        # calculate the order of each node as well as largest node through DFS
        # DFS function
        def traverse_dfs(idx, child_dict, order_dict):
            if idx not in child_dict:
                return 

            for child_idx in child_dict[idx]:
                order_dict[child_idx] = order_dict[idx] + 1
                traverse_dfs(child_idx, child_dict, order_dict)

        # Firstly, for topology analysis, we must firstly merge unifurcation nodes
        

        order_dict = {}
        order_dict[self.idx_soma] = 0
        traverse_dfs(self.idx_soma, self.child_dict, order_dict)
        self.order_dict = order_dict

    def get_num_branches(self):
        return len(self.tree) - 1

    def get_topo_width(self):
        """
        Reference to paper: 
            'Modelling brain-wide neuronal morphology via rooted Cayley trees'
        """
        if not hasattr(self, 'order_dict'):
            self.calc_order_dict()

        order_freq_dict = {}
        multifurcation = self.multifurcation | self.bifurcation
        for idx, order in self.order_dict.items():
            if order not in order_freq_dict:
                order_freq_dict[order] = 0

            if idx == self.p_soma:
                order_freq_dict[order] = 1
                continue
            elif idx in multifurcation:
                order_freq_dict[order] += 1
        self.order_freq_dict = order_freq_dict
        self.topo_width = max(order_freq_dict.values())

        return self.topo_width

    def get_topo_depth(self):
        if not hasattr(self, 'order_dict'):
            self.calc_order_dict()
        self.topo_depth = max(self.order_dict.values())
        return self.topo_depth
        

if __name__ == '__main__':
    from swc_handler import parse_swc, write_swc

    swcfile = '/media/lyf/storage/seu_mouse/swc/xy1z1/18455_00159.swc'
    tree = parse_swc(swcfile)
    morph = Morphology(tree, p_soma=-1)
    morph.get_critical_points()
    new_tree = morph.convert_to_topology_tree()
    
    topo = Topology(new_tree, p_soma=-1)
    #import ipdb; ipdb.set_trace()
    topo.get_topo_width()
    topo.get_topo_depth()
    print(f'''Topology features:
            #stems: {len(morph.stems)}, 
            #tips: {len(morph.tips)}, 
            #unifurcation: {len(morph.unifurcation)}, 
            #bifurcation: {len(morph.bifurcation)},
            #multifurcation: {len(morph.multifurcation)},
            #total_length: {morph.calc_total_length():.2f},
            #dx,dy,dz and volume size: {morph.get_volume_size()}
            #stats for connecting nodes distances: {morph.calc_node_distances()}

            #topo_depth: {topo.topo_depth},
            #topo_width: {topo.topo_width},
            #branches: {topo.get_num_branches()}
            ''')


