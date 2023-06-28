#include <cmath>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <qstring.h>
#include "fastmarching_tree.h"

using namespace std;


struct Swc_Swcfile{
    vector<MyMarker*> outswc;
    QString outswc_file;
};

unordered_map<int, vector<int>> get_child_dict(vector<MyMarker *> tree, int p_idx_in_leaf, unordered_map<MyMarker*, int> parent_ind){
    unordered_map<int, vector<int>> child_dict;

    //int count = 0;
    for (int i=0; i<tree.size();i++){
        //int p_idx = ind[tree[i]->parent];
        int p_idx = parent_ind[tree[i]];
//        if (p_idx == -2) {
//            count++;
//        }
        if (child_dict.count(p_idx)) child_dict[p_idx].push_back(i);
        else child_dict[p_idx] = {i};
    }
    //cout<<count<<endl;
    return child_dict;
}

unordered_map<MyMarker*, int> get_parent_ind(unordered_map<MyMarker*, int> ind,vector<MyMarker *> tree){
    unordered_map<MyMarker*, int> parent_ind;
    int parent_id;
    for (int i=0; i<tree.size();i++){
        if (tree[i]->parent == 0) parent_id = -1;
        else parent_id = ind[tree[i]->parent];
        parent_ind[tree[i]] = parent_id;

    }
//    for (auto kv : ind){
//        if (kv.first->parent == 0) parent_id = -1;
//        else parent_id = ind[kv.first->parent];
//        parent_ind[kv.first] = parent_id;
//    }
    return parent_ind;
}

vector<unordered_set<int>> find_tip_furcation_soma(vector<MyMarker *> tree, unordered_map<int, vector<int>> child_dict){
    
    unordered_set<int> all_nodes_indices, has_child_indices;
    unordered_set<int> tips, furcation;
    int idx_soma = -2;
//    for (int i=0; i<tree.size();i++){
//        all_nodes_indices.insert(i);
//        if (ind[tree[i]->parent] == -1) idx_soma = i;
//    }
    for (int i=0; i<tree.size();i++){
        all_nodes_indices.insert(i);
        if (tree[i]->parent == 0) idx_soma = i;
    }
    for (auto kv : child_dict){
        has_child_indices.insert(kv.first);
        if (kv.first == -1 || kv.first == idx_soma) continue;
        if (kv.second.size() >1) furcation.insert(kv.first);
    }
    for (int node : all_nodes_indices){
        if (!has_child_indices.count(node)) tips.insert(node);
    }

    vector<unordered_set<int>> tip_furcation_soma;
    tip_furcation_soma.push_back(tips);
    tip_furcation_soma.push_back(furcation);
    tip_furcation_soma.push_back({idx_soma});
    return tip_furcation_soma;
}

double calc_dist(double x1,double y1,double z1,double x2,double y2,double z2){
    double dist;
    dist = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2));
    return dist;
}

double calc_ESA(double x,double y,double z,vector<MyMarker *> tree){
    double min_dist = 10000;
    for (int i=0; i<tree.size(); i++){
        double cur_x = tree[i]->x;
        double cur_y = tree[i]->y;
        double cur_z = tree[i]->z;
        double cur_dist = calc_dist(x,y,z,cur_x,cur_y,cur_z);
        if (cur_dist < min_dist) min_dist= cur_dist;
    } 
    return min_dist;
}

unordered_map<int,vector<int>> find_likely_artif(vector<MyMarker *> tree, unordered_set<int> likely_artif_tip, unordered_set<int> furcation, double x_thresh, double y_thresh, double z_thresh, float artif_length, int idx_soma, unordered_map<MyMarker*, int> ind){
    unordered_map<int,vector<int>> likely_artif;
    double cur_x, cur_y, cur_z;
    for (int tip : likely_artif_tip){
        cur_x = abs(tree[tip]->x - tree[idx_soma]->x); 
        cur_y = abs(tree[tip]->y - tree[idx_soma]->y);
        cur_z = abs(tree[tip]->z - tree[idx_soma]->z);
        
        if (cur_x > x_thresh || cur_y > y_thresh || cur_z > z_thresh)
            continue;

        int cur_node = tip;
        double cur_length = 0.0;
        vector<int> likely_artif_nodes;
        //if (1) cout<<cur_node<<endl;
        while (!furcation.count(cur_node) and (cur_node != idx_soma)){
            int p_node = ind[tree[cur_node]->parent];
            cur_length += calc_dist(tree[cur_node]->x, tree[cur_node]->y, tree[cur_node]->z, tree[p_node]->x, tree[p_node]->y, tree[p_node]->z);
            if (cur_length > artif_length) break;
            likely_artif_nodes.push_back(cur_node);
            cur_node = p_node;}

        //if (1) cout<<cur_node<<endl;
        if (cur_length > artif_length) continue;
        likely_artif[likely_artif_nodes[likely_artif_nodes.size()-1]] = likely_artif_nodes;

    }
    return likely_artif;
}

bool delete_artif(vector<MyMarker *>& outtree, vector<MyMarker *>& rotated_outtree, unordered_set<int> furcation,unordered_map<int,vector<int>> likely_artif, float dist_thresh, float pos_ratio, unordered_map<MyMarker*, int>& parent_ind){
    //unordered_map<int, vector<int>> child_dict;
    //unordered_map<MyMarker*, int> parent_ind;
    //parent_ind = get_parent_ind(ind);
    for (auto kv : likely_artif){
        vector<int> likely_artif_nodes = kv.second;
        float pos_num = 0,  num = likely_artif_nodes.size();
        for (int node : likely_artif_nodes){
            double esa  = calc_ESA(outtree[node]->x, outtree[node]->y, outtree[node]->z, rotated_outtree);
            if (esa <= dist_thresh) pos_num++;}
        float cur_ratio = pos_num/num;
        if (cur_ratio < pos_ratio){
                //cout<<"ind:"<<parent_ind[outtree[kv.first]]<<endl;
                //ind[outtree[kv.first]->parent] = -2;
                parent_ind[outtree[kv.first]] = -2;
//                static int kk =0;
//                cout<<++kk<<endl;
//                vector<int> child;
//                child_dict = get_child_dict(outtree,-1, parent_ind);
//                child = child_dict[-2];
//                cout<<"child:"<<child.size()<<endl;
            } 

    }
    return true;
}

//void judging_connect_soma(int idx, unordered_map<int,bool>& temp_tree, unordered_map<int, vector<int>> child_dict){
//    temp_tree[idx] = true;
//    if (child_dict.count(idx) == 0) return;
//    for (int new_idx : child_dict[idx])
//        judging_connect_soma(new_idx, temp_tree, child_dict);

//    return;
//}

void judging_connect_soma(int idx, unordered_map<int,bool>& temp_tree, unordered_map<int, vector<int>> child_dict){
    //temp_tree[idx] = true;
    queue<int> que;
    que.push(idx);
    //int count = 0;
    while (!que.empty()){
        //count++;
//        if (count % 200 == 0)
//            cout<<"count::"<<count<<endl;
        int node = que.front();
        que.pop();
        temp_tree[node] = true;
        for (int i=0; i<child_dict[node].size();i++){
            que.push(child_dict[node][i]);
        }
    }
}


vector<MyMarker *> prune_notConnenctToSoma(vector<MyMarker *>& tree, unordered_map<int, vector<int>> child_dict, int idx_soma){
    unordered_map<int,bool> temp_tree;
    unordered_map<int, MyMarker*> dict_tree;
    for (int i=0; i<tree.size(); i++){
        bool connect_tosoma = false;
        temp_tree[i] = connect_tosoma;
        dict_tree[i] = tree[i];
    }
    
    judging_connect_soma(idx_soma, temp_tree, child_dict);
    int length = tree.size(), num=0;
    for (auto kv : temp_tree){
        if (!kv.second){
            num++;
            //tree.erase(tree.begin() + kv.first);
            dict_tree.erase(kv.first);
        }
    }
    vector<MyMarker *> pruned_tree;
    for(auto kv : dict_tree){
        pruned_tree.push_back(kv.second);
    }
    cout<<"all_nodes:"<<length<<"artifact pruned:"<<num<<endl;
    return pruned_tree;
    
}


vector<MyMarker *> consensus(vector<MyMarker *>& outtree, vector<MyMarker *>& rotated_outtree, V3DLONG* in_sz){
    float region = 1.0/10.0;
    float artif_length = 100.0;
    float dist_thresh = 3;
    float pos_ratio = 0.9;
    
    double x_thresh = in_sz[0] * region, y_thresh = in_sz[1] * region, z_thresh = in_sz[2] * region;

    artif_length /= (1024.0/in_sz[0]);

    unordered_map<MyMarker*, int> ind;
    unordered_map<MyMarker*, int> parent_ind;

    for(int i = 0; i < outtree.size(); i++) ind[outtree[i]] = i;
    parent_ind = get_parent_ind(ind,outtree);

    unordered_map<int, vector<int>> child_dict;
    child_dict = get_child_dict(outtree,-1, parent_ind);

    vector<unordered_set<int>> tip_furcation_soma = find_tip_furcation_soma(outtree, child_dict);
    unordered_set<int> likely_artif_tip = tip_furcation_soma[0];
    unordered_set<int> furcation = tip_furcation_soma[1];
    int idx_soma;
    //for (int s : tip_furcation_soma[2]) int idx_soma = s;
    idx_soma = *(tip_furcation_soma[2].begin());
    //1. find likely_artifacts 
    unordered_map<int,vector<int>> likely_artif = find_likely_artif(outtree,likely_artif_tip, furcation, x_thresh, y_thresh, z_thresh, artif_length, idx_soma, ind);
    //2.consensus(take the intersection)
    //vector<int> child;
//    child_dict = get_child_dict(outtree,-1, parent_ind);
    //child = child_dict[-2];

    delete_artif(outtree, rotated_outtree, furcation, likely_artif, dist_thresh, pos_ratio, parent_ind);

    // update child_dict
    //vector<int> child;
    child_dict = get_child_dict(outtree,-1, parent_ind);
    //child = child_dict[-2];
//    for (int i=0; i<child.size();i++) cout<<child[i]<<endl;
    //3. prune not connect to soma
    vector<MyMarker *> pruned_tree;
    pruned_tree = prune_notConnenctToSoma(outtree, child_dict, idx_soma);

    return pruned_tree;
}


bool affine_swc(vector<MyMarker *>& rotated_outtree, double a, double b, double c, double d, V3DLONG* in_sz){
    V3DLONG D = in_sz[2], H = in_sz[1], W = in_sz[0];
    for (int i=0; i<rotated_outtree.size(); i++){
        double x = rotated_outtree[i]->x, y = rotated_outtree[i]->y;
        x -= V3DLONG(W/2);
        y -= V3DLONG(H/2);
        double new_x = x * a + y * c;
        double new_y = x * b + y * d;
        new_x += V3DLONG(W/2);
        new_y += V3DLONG(H/2);
        
        if (new_x > 0 && new_x < W && new_y > 0 && new_y < H)
            rotated_outtree[i]->x = new_x;
            rotated_outtree[i]->y = new_y; 
    }
    return true; 
}


unsigned char* affine_img(unsigned char* img, float a, float b, float c, float d, V3DLONG* in_sz){
    V3DLONG D = in_sz[2], H = in_sz[1], W = in_sz[0];
    V3DLONG center[2] = {int(H/2),int(W/2)};
    float transform_x[3] = {a, b, 0};
    float transform_y[3] = {c, d, 0};
    float transform_z[3] = {0, 0, 1};
    unsigned char *rotated_img = new unsigned char[D*H*W];//[536870919];

    V3DLONG loc1,loc2;
    //int count = 0;

    for (V3DLONG z=0; z<D; z++){
        for (V3DLONG y=0; y<H; y++){
            for (V3DLONG x=0; x<W; x++){
                    V3DLONG offset = z * H * W;
                    V3DLONG src_pos[3] = {x-center[1],y-center[0],z};
                    V3DLONG src_x = src_pos[0]*transform_x[0] + src_pos[1]*transform_x[1] + src_pos[2]*transform_x[2];
                    V3DLONG src_y = src_pos[0]*transform_y[0] + src_pos[1]*transform_y[1] + src_pos[2]*transform_y[2];
                    V3DLONG src_z = src_pos[0]*transform_z[0] + src_pos[1]*transform_z[1] + src_pos[2]*transform_z[2];
                    src_x += center[1];
                    src_y += center[0];

                    if (src_x >= W || src_y >= H || src_x <0 || src_y <0){
                        rotated_img[offset + y*W + x] = 1;
                    }
                    else{
                        rotated_img[offset + y*W + x] = img[offset + src_y*W + src_x];
                        //rotated_img_v.push_back(img[offset + src_y*W + src_x]);
                    }
                            
            }
        }
    }

    return rotated_img;
}

unsigned char* rotation_image(unsigned char* img, V3DLONG* in_sz){
    float pi = 3.14159;
    // Affine
    float A = 45.0;
    float theta = - pi * A / 180;
    float a = cos(theta), b=sin(theta), c=-sin(theta), d=cos(theta);
    unsigned char* rotated_img = 0;
     
    rotated_img = affine_img(img, a, b, c, d, in_sz);
    return rotated_img; 

}

bool rotation_swc(vector<MyMarker *>& rotated_outtree, V3DLONG* in_sz){
    float pi = 3.14159;
    // Affine
    float A = 45.0;
    float theta = - pi * A / 180;
    double a = cos(theta), b=-sin(theta), c=sin(theta), d=cos(theta);

    return affine_swc(rotated_outtree, a, b, c, d, in_sz);
}

