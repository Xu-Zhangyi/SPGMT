# Structure and Position-aware Graph Modeling for Trajectory Similarity Computation over Road Networks

Source codes for Structure and Position-aware Graph Modeling for Trajectory Similarity Computation over Road Networks

## Running Procedures:

1. Extract the dataset folder, you can get the 'matching_result.pt', 'node.csv', and 'edge_weight.csv' of Beiijng or Porto dataset
2. Run 'spatial_preprocess.py' to obtain the initial structural embeddings for trajectories as well as data for 'shuffle_node_list.npy', 'shuffle_coor_list.npy'.
3. Run 'spatial_similarity_computation.py' to compute the pairwise point distances and ground truth similarities for trajectories.
4. Run 'generate_node_knn.py' to get the kNN neighbors for each node in the road network.
5. Run 'spatial_data_utils.py' to obtain the anchor node for SPGMT.
6. Run 'main.py' to train SPGMT. To test SPGMT, you can load the saved model and use the 'Spa_eval' function in the 'main.py'. The experiments on test data, long trajectories and scalibility study are all included in this function. In addition, you can modify the parameters in 'setting.py' to run the model, and 'usePI', 'useSI', 'useGI' correspond to our three variants in the ablation study.
