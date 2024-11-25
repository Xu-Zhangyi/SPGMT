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
    for i in tqdm(range(len(input_dis_matrix))):
        input_r = np.array(input_dis_matrix[i])
        # input_r = input_r[np.where(input_r != -1)[0]]
        idx = np.argsort(input_r)
        # label.append(idx[1])
        val = input_r[idx]
        idx = idx[val != -1]
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
re_test_y = get_label(re_test_dis_matrix, 50)  # 29894

out_test_list = []
out_coor_list = []
out_y = []
max_len = 0
for i in range(11):
    for j in range(len(re_test_list)):
        if len(re_test_list[j]) <= 1600:
            # if max_len<len(re_test_list[j]):
            #     max_len=len(re_test_list[j])
            out_test_list.append(re_test_list[j])
            out_coor_list.append(re_test_coor[j])
            out_y.append(re_test_y[j])
# print(max_len)#beijing: 3625 porto:1550
out_test_list = np.array(out_test_list, dtype=object)
out_coor_list = np.array(out_coor_list, dtype=object)
out_y = np.array(out_y, dtype=object)
np.savez(config.spatial_num_test_set, vali_list=out_test_list, vali_coor=out_coor_list, vali_y=out_y)
