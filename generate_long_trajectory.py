import numpy as np
import os.path as osp
from tqdm import tqdm
from setting import SetParameter
import random
config = SetParameter()
random.seed(1933)
np.random.seed(1933)
if config.dataset == 'beijing':
    point_num = 112557  # tdrive: 74671; beijing: 112557; porto: 128466
    matrix_num = 113000  # tdrive: 75000; beijing: 113000; porto: 129000
    LCRS_num = 7250
elif config.dataset == 'porto':
    point_num = 128466
    matrix_num = 129000
    LCRS_num = 3286


# if str(config.dataset) == "tdrive" or "beijing" or "porto":
#     if str(config.distance_type) == "TP":
#         self.extra_coe = 4
#     elif str(config.distance_type) == "DITA":
#         if str(config.dataset) == "beijing":
#             self.extra_coe = 8
#         if str(config.dataset) == "porto":
#             self.extra_coe = 4
#     elif str(config.distance_type) == "LCRS":
#         self.extra_coe = 16
#     elif str(config.distance_type) == "discret_frechet":
#         self.extra_coe = 4


if str(config.dataset) == "tdrive" or "beijing" or "porto":
    if str(config.distance_type) == "TP":
        coe = 8*4
    elif str(config.distance_type) == "DITA":
        if str(config.dataset) == "beijing":
            coe = 32*8*2
        if str(config.dataset) == "porto":
            coe = 32*4
    elif str(config.distance_type) == "LCRS":
        coe = 4*16
    elif str(config.distance_type) == "discret_frechet":
        coe = 8*4


def re_matrix(traj_list, coor_list, input_dis_matrix, l):
    idx = []
    for i in tqdm(range(len(input_dis_matrix))):
        if np.sum(input_dis_matrix[i] >= 0) > l:
            idx.append(i)
    idx = np.array(idx)
    out_traj_list = traj_list[idx]
    out_coor_list = coor_list[idx]
    out_dis_matrix = input_dis_matrix[idx.reshape(-1, 1), idx.reshape(1, -1)]

    return out_traj_list, out_coor_list, out_dis_matrix


def get_label(input_dis_matrix, count):
    label = []
    for i in tqdm(range(len(input_dis_matrix))):  # (5000,5000)
        input_r = np.array(input_dis_matrix[i])
        # input_r = input_r[np.where(input_r != -1)[0]]
        idx = np.argsort(input_r)
        # label.append(idx[1])
        val = input_r[idx]
        idx = idx[val != -1]  # 升序,第0位是自己
        label.append(idx[1:count+1])
    return np.array(label)


def get_train_label(input_dis_matrix, count):
    label = []
    neg_label = []

    label_dis = []
    neg_label_dis = []
    for i in tqdm(range(len(input_dis_matrix))):  # (5000,5000)
        input_r = np.array(input_dis_matrix[i])
        # input_r = input_r[np.where(input_r != -1)[0]]
        idx = np.argsort(input_r)
        # label.append(idx[1])
        val = input_r[idx]
        re_idx = idx[val != -1]
        re_val = val[val != -1]

        label.append(re_idx[1:count+1])
        label_dis.append(re_val[1:count+1])
        neg_label.append(re_idx[count+1:])
        neg_label_dis.append(re_val[count+1:])

    label = np.array(label, dtype=object)
    neg_label = np.array(neg_label, dtype=object)
    label_dis = np.array(label_dis, dtype=object)
    neg_label_dis = np.array(neg_label_dis, dtype=object)
    return label, neg_label, label_dis, neg_label_dis


def get_long_traj(traj_list, coor_list, input_dis_matrix, l):
    idx = []
    for i in tqdm(range(len(traj_list))):
        if len(traj_list[i]) >= l:
            idx.append(i)
    idx = np.array(idx)
    out_traj_list = traj_list[idx]
    out_coor_list = coor_list[idx]
    out_dis_matrix = input_dis_matrix[idx.reshape(-1, 1), idx.reshape(1, -1)]
    return out_traj_list, out_coor_list, out_dis_matrix


all_node_list_int = np.load(osp.join('data', config.dataset, 'st_traj', 'shuffle_node_list.npy'), allow_pickle=True)
all_coor_list_int = np.load(osp.join('data', config.dataset, 'st_traj', 'shuffle_coor_list.npy'), allow_pickle=True)
print(config.dataset)
print(config.distance_type)

test_list = all_node_list_int[20000:50000]
test_coor = all_coor_list_int[20000:50000]


test_dis_matrix = np.load(osp.join('ground_truth', config.dataset,
                          config.distance_type, 'test_spatial_distance_50000.npy'))
np.fill_diagonal(test_dis_matrix, 0)

re_test_list, re_test_coor, re_test_dis_matrix = re_matrix(test_list, test_coor, test_dis_matrix, 50)
long_test_list, long_test_coor, long_test_dis_matrix = get_long_traj(
    re_test_list, re_test_coor, re_test_dis_matrix, 150)  
long_test_y = get_label(long_test_dis_matrix, 50)  

np.savez(config.spatial_long_test_set, vali_list=long_test_list, vali_coor=long_test_coor, vali_y=long_test_y)  # 30000-106
