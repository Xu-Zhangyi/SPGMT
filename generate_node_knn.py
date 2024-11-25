import numpy as np
from tqdm import tqdm

city = 'porto'

if city == 'beijing':
    point_num = 112557  # tdrive: 74671; beijing: 112557; porto: 128466
    matrix_num = 113000  # tdrive: 75000; beijing: 113000; porto: 129000
elif city == 'porto':
    point_num = 128466
    matrix_num = 129000
else:
    point_num = 74671
    matrix_num = 75000
# extract 50-nn and distances for each node
point_dis = np.load(f'/data/xuzhangyi/SPGMT/{city}/Point_dis_matrix.npy')
for i in range(len(point_dis)):
    point_dis[i][i] = 0.0

knn_neighbor = []
knn_distance = []
max_cnt = 0
for i in tqdm(range(point_num)):
    # sorted_zero_id = np.argwhere(np.argsort(point_dis[i]) == i)[0][0]
    sorted_list = np.argsort(point_dis[i])  # 距离由小到大排列的索引
    sorted_zero_id = np.argwhere(sorted_list == i)[0][0]  # 即距离为0的位置
    tmp_knn_neighbor = []
    tmp_knn_distance = []
    if sorted_zero_id < (matrix_num - 50):
        # for j in range(1, 51):
        #     tmp_knn_neighbor.append(sorted_list[sorted_zero_id + j])
        #     tmp_knn_distance.append(point_dis[i][sorted_list[sorted_zero_id + j]])
        # knn_neighbor.append(tmp_knn_neighbor)
        # knn_distance.append(tmp_knn_distance)
        knn_neighbor.append(sorted_list[sorted_zero_id + 1:sorted_zero_id + 50])
        knn_distance.append(point_dis[i][sorted_list[sorted_zero_id + 1:sorted_zero_id + 50]])
    else:
        if (sorted_zero_id - (matrix_num - 50)) > max_cnt:
            max_cnt = sorted_zero_id - (matrix_num - 50)
        # for j in range(1, (matrix_num - sorted_zero_id)):
        #     tmp_knn_neighbor.append(sorted_list[sorted_zero_id + j])
        #     tmp_knn_distance.append(point_dis[i][sorted_list[sorted_zero_id + j]])
        # knn_neighbor.append(tmp_knn_neighbor)
        # knn_distance.append(tmp_knn_distance)
        knn_neighbor.append(sorted_list[sorted_zero_id + 1:matrix_num - 1])
        knn_distance.append(point_dis[i][sorted_list[sorted_zero_id + 1:matrix_num - 1]])

print(max_cnt)
np.save(f'./ground_truth/{city}/knn_neighbor', np.array(knn_neighbor, dtype=object))
np.save(f'./ground_truth/{city}/knn_distance', np.array(knn_distance, dtype=object))

# store edge index for 3 layers
all_node_knn_neighbor = np.load(f'./ground_truth/{city}/knn_neighbor.npy', allow_pickle=True)
all_node_knn_distance = np.load(f'./ground_truth/{city}/knn_distance.npy', allow_pickle=True)

k5_neighbor = []
k5_distance = []
k10_neighbor = []
k10_distance = []
k15_neighbor = []
k15_distance = []
for i in range(point_num):
    i_knn_neighbor = all_node_knn_neighbor[i]
    i_knn_distance = all_node_knn_distance[i]
    for idx, j in enumerate(i_knn_neighbor[0:5]):
        k5_neighbor.append(np.array([i, j]))
        k5_distance.append(i_knn_distance[idx])
    for idx, j in enumerate(i_knn_neighbor[5:10]):
        k10_neighbor.append(np.array([i, j]))
        k10_distance.append(i_knn_distance[idx + 5])
    for idx, j in enumerate(i_knn_neighbor[10:15]):
        k15_neighbor.append(np.array([i, j]))
        k15_distance.append(i_knn_distance[idx + 10])

np.save(f'./dataset/{city}/5/k5_neighbor', np.array(k5_neighbor))
np.save(f'./dataset/{city}/5/k5_distance', np.array(k5_distance))
np.save(f'./dataset/{city}/5/k10_neighbor', np.array(k10_neighbor))
np.save(f'./dataset/{city}/5/k10_distance', np.array(k10_distance))
np.save(f'./dataset/{city}/5/k15_neighbor', np.array(k15_neighbor))
np.save(f'./dataset/{city}/5/k15_distance', np.array(k15_distance))
print('successful!')
