from setting import SetParameter
import torch
import numpy as np
import pandas as pd
from torch_geometric.nn import Node2Vec
import random
from tqdm import tqdm
import ast
import torch.nn as nn
import datetime
from Model import Date2VecConvert
import os
config = SetParameter()
random.seed(1953)


def prepare_dataset(trajfile, kseg=5):
    node_list = torch.load(trajfile)  # beijing:786801
    node_list_int = []
    coor_trajs = []
    print("{0} raw trajectory has {1}.".format(config.dataset, len(node_list)))  # beijing:651554
    for nlist in tqdm(node_list, desc='load trajectory'):
        if len(nlist[0]) >= 10:
            node_list_int.append(nlist[0])
            coor_trajs.append(nlist[1])
    print("{0} trajectory has {1}.".format(config.dataset, len(node_list_int)))
    node_list_int = np.array(node_list_int, dtype=object)
    coor_trajs = np.array(coor_trajs, dtype=object)
    # %=================================================================================================================
    shuffle_index = list(range(len(node_list_int)))
    random.shuffle(shuffle_index)
    shuffle_index = shuffle_index[:config.dataset_size]  # 5w size of dataset
    coor_trajs = coor_trajs[shuffle_index]
    node_list_int = node_list_int[shuffle_index]
    # %=================================================================================================================
    kseg_coor_trajs = []
    for t in tqdm(coor_trajs, desc='computer ksegment'):
        kseg_coor = []
        seg = len(t) // kseg
        t = np.array(t)
        for i in range(kseg):
            if i == kseg - 1:
                kseg_coor.append(np.mean(t[i * seg:], axis=0))
            else:
                kseg_coor.append(np.mean(t[i * seg:i * seg + seg], axis=0))
        kseg_coor_trajs.append(kseg_coor)
    kseg_coor_trajs = np.array(kseg_coor_trajs, dtype=object)
    print("complete: ksegment")
    # %=================================================================================================================
    # kseg_coor_trajs = kseg_coor_trajs[shuffle_index]
    if not os.path.exists(config.data_file):
        os.makedirs(config.data_file)
    
    np.save(str(config.shuffle_coor_file), coor_trajs)
    np.save(str(config.shuffle_node_file), node_list_int)
    np.save(str(config.shuffle_kseg_file), kseg_coor_trajs)
    np.save(str(config.shuffle_index_file), shuffle_index)
    print('successful!')


def prepare_st_dataset(trajfile, timefile, kseg=5):
    df_traj_list = pd.read_csv(trajfile)

    df_traj_list['Node_list'] = df_traj_list['Node_list'].apply(ast.literal_eval)
    df_traj_list['Coor_list'] = df_traj_list['Coor_list'].apply(ast.literal_eval)

    node_list = df_traj_list['Node_list'].tolist()
    coor_list = df_traj_list['Coor_list'].tolist()
    print("{0} raw trajectory has {1}.".format(config.dataset, len(node_list)))  # beijing:651554

    df_time_list = pd.read_csv(timefile)
    df_time_list['Time_list'] = df_time_list['Time_list'].apply(ast.literal_eval)
    time_list = df_time_list['Time_list'].tolist()
    print("Time List:", time_list[0])

    node_list = np.array(node_list, dtype=object)
    coor_list = np.array(coor_list, dtype=object)
    time_list = np.array(time_list, dtype=object)

    kseg_coor_trajs = []
    for t in coor_list:
        kseg_coor = []
        seg = len(t) // kseg
        t = np.array(t)
        for i in range(kseg):
            if i == kseg - 1:
                kseg_coor.append(np.mean(t[i * seg:], axis=0))
            else:
                kseg_coor.append(np.mean(t[i * seg:i * seg + seg], axis=0))
        kseg_coor_trajs.append(kseg_coor)
    kseg_coor_trajs = np.array(kseg_coor_trajs)
    print("complete: ksegment")

    # %=================================================================================================================
    shuffle_index = list(range(len(node_list)))
    random.shuffle(shuffle_index)
    shuffle_index = shuffle_index[:config.dataset_size]

    node_list = node_list[shuffle_index]
    coor_list = coor_list[shuffle_index]
    time_list = time_list[shuffle_index]
    kseg_coor_trajs = kseg_coor_trajs[shuffle_index]

    # %=================================================================================================================
    # kseg_coor_trajs = kseg_coor_trajs[shuffle_index]
    np.save(str(config.shuffle_coor_file), coor_list)
    np.save(str(config.shuffle_node_file), node_list)
    np.save(str(config.shuffle_time_file), time_list)
    np.save(str(config.shuffle_kseg_file), kseg_coor_trajs)
    np.save(str(config.shuffle_index_file), shuffle_index)
    print('successful!')
# %==============================================================================================================================


class Date2vec(nn.Module):
    def __init__(self):
        super(Date2vec, self).__init__()
        self.d2v = Date2VecConvert(model_path="./d2v_model/d2v_98291_17.169918439404636.pth")

    def forward(self, time_seq):
        all_list = []
        for one_seq in tqdm(time_seq):
            one_list = []
            for timestamp in one_seq:
                t = datetime.datetime.fromtimestamp(timestamp)
                t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
                x = torch.Tensor(t).float()
                embed = self.d2v(x)
                one_list.append(embed)

            one_list = torch.cat(one_list, dim=0)
            one_list = one_list.view(-1, 64)

            all_list.append(one_list.numpy().tolist())

        all_list = np.array(all_list, dtype=object)

        return all_list
# %==============================================================================================================================


def read_graph():
    print(f'preprocess:{config.dataset}')
    edge = str(config.edge_file)
    node = str(config.node_file)

    df_edge = pd.read_csv(edge, sep=',')
    df_node = pd.read_csv(node, sep=',')

    edge_index = df_edge[["s_node", "e_node"]].to_numpy()
    num_node = df_node["node"].size

    print("{0} road network has {1} edges.".format(config.dataset, edge_index.shape[0]))  # beijing:249097,tdrive:165280
    print("{0} road network has {1} nodes.".format(config.dataset, num_node))  # beijing:112557, tdrive:74671

    return edge_index, num_node


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(config.device), neg_rw.to(config.device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch(model, loader, optimizer):
    last_loss = 1
    print("Training node embedding with node2vec...")
    for i in range(100):
        loss = train(model, loader, optimizer)
        print('Epoch: {0} \tLoss: {1:.4f}'.format(i, loss))
        if abs(last_loss - loss) < 1e-5:
            break
        else:
            last_loss = loss


@torch.no_grad()
def save_embeddings(model, num_nodes, dataset, device):
    model.eval()
    node_features = model(torch.arange(num_nodes, device=device)).cpu().numpy()
    np.save("data/" + dataset + "/node_features.npy", node_features)
    print("Node embedding saved at: data/" + dataset + "/node_features.npy")
    return


if __name__ == "__main__":
    edge_index, num_node = read_graph()

    feature_size = config.feature_size
    walk_length = config.node2vec["walk_length"]
    context_size = config.node2vec["context_size"]
    walks_per_node = config.node2vec["walks_per_node"]
    p = config.node2vec["p"]
    q = config.node2vec["q"]
    edge_index = torch.LongTensor(edge_index).t().contiguous().to(config.device)

    model = Node2Vec(
        edge_index,
        embedding_dim=feature_size,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=1,
        p=p,
        q=q,
        sparse=True,
        num_nodes=num_node
    ).to(config.device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
    train_epoch(model, loader, optimizer)
    save_embeddings(model, num_node, str(config.dataset), config.device)

    if config.dataset == 'beijing' or 'porto':
        prepare_dataset(trajfile=str(config.traj_file), kseg=config.kseg)
    elif config.dataset == 'tdrive':
        prepare_st_dataset(trajfile=str(config.tdrive_traj_file),
                           timefile=str(config.tdrive_time_file),
                           kseg=config.kseg)
        d2vec = Date2vec()
        timelist = np.load(str(config.shuffle_time_file), allow_pickle=True)
        d2v = d2vec(timelist)
        print(len(d2v))
        np.save(str(config.shuffle_d2vec_file), d2v)
