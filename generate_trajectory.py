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
        idx = idx[val != -1]
        if len(idx) < 51:
            print(idx)
        label.append(idx[1:count+1])
    return np.array(label)


def get_train_label(input_dis_matrix, count):
    label = []
    neg_label = []

    label_dis = []
    neg_label_dis = []
    for i in tqdm(range(len(input_dis_matrix))):
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


all_node_list_int = np.load(osp.join('data', config.dataset, 'st_traj', 'shuffle_node_list.npy'), allow_pickle=True)
all_coor_list_int = np.load(osp.join('data', config.dataset, 'st_traj', 'shuffle_coor_list.npy'), allow_pickle=True)
print(config.dataset)
print(config.distance_type)
train_list = all_node_list_int[0:config.train_set_size]
vali_list = all_node_list_int[config.train_set_size:config.vali_set_size]
test_list = all_node_list_int[config.vali_set_size:config.test_set_size]


train_coor = all_coor_list_int[0:config.train_set_size]
vali_coor = all_coor_list_int[config.train_set_size:config.vali_set_size]
test_coor = all_coor_list_int[config.vali_set_size:config.test_set_size]


train_dis_matrix = np.load(osp.join('ground_truth', config.dataset,
                           config.distance_type, 'train_spatial_distance_50000.npy'))
vali_dis_matrix = np.load(osp.join('ground_truth', config.dataset,
                          config.distance_type, 'vali_spatial_distance_50000.npy'))
test_dis_matrix = np.load(osp.join('ground_truth', config.dataset,
                          config.distance_type, 'test_spatial_distance_50000.npy'))

np.fill_diagonal(train_dis_matrix, 0)
np.fill_diagonal(vali_dis_matrix, 0)
np.fill_diagonal(test_dis_matrix, 0)
if config.distance_type == 'LCRS':
    train_dis_matrix[train_dis_matrix == LCRS_num] = -1
re_train_list, re_train_coor, re_train_dis_matrix = re_matrix(
    train_list, train_coor, train_dis_matrix, config.pos_num*2)
re_vali_list, re_vali_coor, re_vali_dis_matrix = re_matrix(vali_list, vali_coor, vali_dis_matrix, 50)
re_test_list, re_test_coor, re_test_dis_matrix = re_matrix(test_list, test_coor, test_dis_matrix, 50)


norm_num = np.max(re_train_dis_matrix)
re_train_dis_matrix = re_train_dis_matrix / norm_num * config.coe
re_train_dis_matrix[re_train_dis_matrix < 0] = -1

train_y, train_neg_y, train_dis, train_neg_dis = get_train_label(re_train_dis_matrix, config.pos_num)
vali_y = get_label(re_vali_dis_matrix, 50)
test_y = get_label(re_test_dis_matrix, 50)

np.savez(config.spatial_train_set,
         train_list=re_train_list,
         train_coor=re_train_coor,
         train_y=train_y,
         train_neg_y=train_neg_y,
         train_dis=train_dis,
         train_neg_dis=train_neg_dis,
         coe=config.coe)
np.savez(config.spatial_vali_set, vali_list=re_vali_list, vali_coor=re_vali_coor, vali_y=vali_y)
np.savez(config.spatial_test_set, vali_list=re_test_list, vali_coor=re_test_coor, vali_y=test_y)
