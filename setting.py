import logging
import os.path as osp
import torch
import os


class SetParameter:
    def __init__(self):
        self.dataset = 'beijing'  # beijing,porto,tdrive
        self.distance_type = 'TP'  # 'TP','DITA','discret_frechet','LCRS','NetERP'

        if self.dataset == 'tdrive':
            self.dataset_size = 30000
            self.train_set_size = 10000
            self.vali_set_size = 14000
            self.test_set_size = 30000
        elif self.dataset == 'beijing' or self.dataset == 'porto':
            self.dataset_size = 50000  # tdrive: 30000; beijing, porto: 50000
            self.train_set_size = 15000  # tdrive: 10000; beijing, porto: 15000
            self.vali_set_size = 20000  # tdrive: 14000; beijing, porto: 20000
            self.test_set_size = 50000  # tdrive: 30000; beijing, porto: 50000

        self.kseg = 5
        self.pos_num = 10

        self.node_file = osp.join('dataset', self.dataset, 'node.csv')
        self.edge_file = osp.join('dataset', self.dataset, 'edge_weight.csv')
        self.traj_file = osp.join('dataset', self.dataset, 'matching_result.pt')

        self.tdrive_traj_file = osp.join('dataset', self.dataset, 'matching_result.csv')
        self.tdrive_time_file = osp.join('dataset', self.dataset, 'time_drop_list.csv')

        self.data_file = osp.join('data', self.dataset, 'st_traj')
        self.shuffle_node_file = osp.join(self.data_file, 'shuffle_node_list.npy')
        self.shuffle_coor_file = osp.join(self.data_file, 'shuffle_coor_list.npy')
        self.shuffle_kseg_file = osp.join(self.data_file, 'shuffle_kseg_list.npy')
        self.shuffle_index_file = osp.join(self.data_file, 'shuffle_index_list.npy')
        self.shuffle_time_file = osp.join(self.data_file, 'shuffle_time_list.npy')

        self.shuffle_d2vec_file = osp.join(self.data_file, 'shuffle_d2vec_list.npy')

        self.spatial_train_set = osp.join('ground_truth', self.dataset, self.distance_type, 'spatial_train_set.npz')
        self.spatial_vali_set = osp.join('ground_truth', self.dataset, self.distance_type, 'spatial_vali_set.npz')
        self.spatial_test_set = osp.join('ground_truth', self.dataset, self.distance_type, 'spatial_test_set.npz')
        self.spatial_long_test_set = osp.join('ground_truth', self.dataset,
                                              self.distance_type, 'spatial_long_test_set.npz')
        self.spatial_num_test_set = osp.join('ground_truth', self.dataset,
                                             self.distance_type, 'spatial_num_test_set.npz')

        # The number of nodes in the network
        self.pointnum = {
            'beijing': 113000,
            'porto': 129000,
            'tdrive': 75000
        }
        self.truenum = {
            'beijing': 112557,
            'porto': 128466,
            'tdrive': 74671
        }

        self.alpha1 = 0.75  # 0.55,0.65,0.75,0.85,0.95
        self.alpha2 = 0.25  # 0.45,0.35,0.25,0.15,0.05

        self.num_knn = 5  # 3,4,5,6,7

        self.save_folder = None
        self.feature_size = 64  # node2vec feature size
        self.embedding_size = 64  # GNN embedding size
        self.date2vec_size = 64  # TRM hidden size
        self.hidden_size = 128
        self.num_layers = 1
        self.dropout_rate = 0
        self.learning_rate = 0.001
        self.concat = False
        self.epochs = 150
        self.early_stop = 75

        self.gtraj = {
            'train_batch': 20,
            'test_batch': 128,
            'usePE': True,
            'useSI': True,
            'useLSTM': True
        }

        self.node2vec = {
            'walk_length': 20,
            'context_size': 10,
            'walks_per_node': 10,
            'num_neg_samples': 1,
            'p': 1,
            'q': 1
        }
        self.cuda = str('0')
        self.device = torch.device('cuda:' + self.cuda) if torch.cuda.is_available() else torch.device('cpu')

        if str(self.distance_type) == "TP":
            self.coe = 8*4
        elif str(self.distance_type) == "DITA":
            if str(self.dataset) == "beijing":
                self.coe = 32*8*2
            if str(self.dataset) == "porto":
                self.coe = 32*4
        elif str(self.distance_type) == "LCRS":
            self.coe = 4*16
        elif str(self.distance_type) == "discret_frechet":
            self.coe = 8*2

        allowed_dis = {'TP', 'DITA', 'discret_frechet', 'LCRS', 'NetERP'}
        if self.distance_type not in allowed_dis:
            raise ValueError(f"Invalid value for distance_type. Allowed values are: {allowed_dis}")
        allowed_data = {'beijing', 'porto', 'tdrive'}
        if self.dataset not in allowed_data:
            raise ValueError(f"Invalid value for distance_type. Allowed values are: {allowed_data}")


def setup_logger(fname=None):

    if not logging.root.hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=fname,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)
