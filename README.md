# SPGMT

Source codes for SPGMT

## Running Procedures:

1. Extract the dataset folder, you can get the 'matching_result.pt', 'node.csv', and 'edge_weight.csv' of Beijing or Porto dataset.
   
   Example: `dataset/beijing/matching_result.pt`

2. Run 'spatial_preprocess.py' to obtain the initial structural embeddings 'node_features.npy' for trajectories as well as 'shuffle_node_list.npy', 'shuffle_coor_list.npy'.
   
   Example: `data/beijing/node_features.npy`, `data/beijing/st_traj/shuffle_node_list.npy`

3. Run 'spatial_similarity_computation.py' to compute the pairwise point distances and ground truth similarities for trajectories. This will take some time.
    
    Example: `ground_truth/beijing/Point_dis_matrix.npy`, `ground_truth/beijing/TP/train_spatial_distance_50000.npy`

4. Run 'generate_trajectory.py' to obtain the training set, validation set, and test set.

    Example: `ground_truth/beijing/TP/spatial_train_set.npz`

5. Run 'generate_node_knn.py' to get the kNN neighbors for each node in the road network.

    Example: `dataset/beijing/5/k5_neighbor.npy`

6. Run 'spatial_data_utils.py' to obtain the position embeddings for SPGMT.

    Example: `dataset/beijing/distance_to_anchor_node.pt`

7. Run 'main.py' to train SPGMT. To test SPGMT, you can load the saved model and use the 'Spa_eval' function in the 'main.py'. The experiments on test data, long trajectories and scalibility study are all included in this function. In addition, you can modify the parameters in 'setting.py' to run the model, and 'usePI', 'useSI', 'useGI' correspond to our variants in the ablation study.
