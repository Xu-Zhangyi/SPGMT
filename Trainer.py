from setting import SetParameter
from model_network import GraphTrajSTEncoder, GraphTrajSimEncoder
import spatial_data_utils
import torch
from lossfun import SpaLossFun
import time
from tqdm import tqdm
import numpy as np
import test_method
import logging
import os.path as osp
import os


class GTrajsim_Trainer(object):
    def __init__(self, config):

        self.feature_size = config.feature_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate
        self.concat = config.concat
        self.device = "cuda:" + str(config.cuda)
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs

        self.train_batch = config.gtraj["train_batch"]
        self.test_batch = config.gtraj["test_batch"]
        self.traj_file = str(config.traj_file)
        self.usePE = config.gtraj["usePE"]
        self.useSI = config.gtraj["useSI"]
        self.useLSTM = config.gtraj["useLSTM"]

        self.dataset = str(config.dataset)
        self.distance_type = str(config.distance_type)
        self.early_stop = config.early_stop
        self.alpha1 = config.alpha1
        self.alpha2 = config.alpha2
        self.num_knn = config.num_knn
        self.config = config
        # %========================================
        # self.train_data_loader = spatial_data_utils.train_data_loader(config.spatial_train_set, self.train_batch)
        # self.vali_data_loader, self.vali_lenth = spatial_data_utils.vali_data_loader(config.spatial_vali_set, self.train_batch)

    def Spa_eval(self, load_model=None, is_long_traj=False):
        logging.info('SPGMT on ' + self.dataset + ' with ' + self.distance_type)
        net = GraphTrajSimEncoder(feature_size=self.feature_size,
                                  embedding_size=self.embedding_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  dropout_rate=self.dropout_rate,
                                  concat=self.concat,
                                  device=self.device,
                                  usePE=self.usePE,
                                  useSI=self.useSI,
                                  useLSTM=self.useLSTM,
                                  dataset=self.dataset,
                                  alpha1=self.alpha1,
                                  alpha2=self.alpha2,)
        if load_model != None:
            logging.info(load_model)
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)
            if is_long_traj:
                test_data_loader, test_lenth = spatial_data_utils.vali_data_loader(
                    self.config.spatial_long_test_set, self.test_batch)
            else:
                test_data_loader, test_lenth = spatial_data_utils.vali_data_loader(
                    self.config.spatial_test_set, self.test_batch)

            logging.info(f'test size:{test_lenth}')
            if self.useSI:
                road_network = spatial_data_utils.load_neighbor(self.dataset, self.num_knn)
                for item in road_network:
                    item = item.to(self.device)
            else:
                road_network = spatial_data_utils.load_netowrk(self.dataset).to(self.device)
            distance_to_anchor_node = spatial_data_utils.load_region_node(self.dataset).to(self.device)

            net.eval()
            with torch.no_grad():
                start_test_epoch = time.time()
                if self.useLSTM:
                    test_embedding = torch.zeros((test_lenth, self.hidden_size*2),
                                                 device=self.device, requires_grad=False)
                else:
                    test_embedding = torch.zeros((test_lenth, self.hidden_size),
                                                 device=self.device, requires_grad=False)
                test_label = torch.zeros((test_lenth, 50), requires_grad=False, dtype=torch.long)

                for batch in tqdm(test_data_loader):
                    (data, coor, label, data_length, idx) = batch
                    a_embedding = net([road_network, distance_to_anchor_node], data, coor, data_length)
                    test_embedding[idx] = a_embedding
                    test_label[idx] = label
                end_test_epoch = time.time()
                acc = test_method.test_spa_model(test_embedding, test_label, self.device)
                acc = acc.mean(axis=0)  # HR-10 HR-50 R10@50 R1@1 R1@10 R1@50
                acc[0] = acc[0] / 10.0
                acc[1] = acc[1] / 50.0
                acc[2] = acc[2] / 10.0
                logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                    self.dataset, self.distance_type, test_lenth))
                end_test = time.time()
                logging.info(f"epoch test time: {end_test_epoch - start_test_epoch}")
                logging.info(f"all test time: {end_test - start_test_epoch}")
                logging.info(acc)

    def Spa_train(self):
        logging.info('SPGMT on ' + self.dataset + ' with ' + self.distance_type)
        logging.info('positive num: ' + str(self.config.pos_num))

        logging.info('GPU: '+str(self.config.cuda))
        net = GraphTrajSimEncoder(feature_size=self.feature_size,
                                  embedding_size=self.embedding_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  dropout_rate=self.dropout_rate,
                                  concat=self.concat,
                                  device=self.device,
                                  usePE=self.usePE,
                                  useSI=self.useSI,
                                  useLSTM=self.useLSTM,
                                  dataset=self.dataset,
                                  alpha1=self.alpha1,
                                  alpha2=self.alpha2).to(self.device)

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        milestones_list = [15, 30, 45, 60, 75, 90, 115]
        logging.info(milestones_list)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_list, gamma=0.2)

        lossfunction = SpaLossFun(self.train_batch, self.distance_type).to(self.device)
        distance_to_anchor_node = spatial_data_utils.load_region_node(self.dataset).to(self.device)

        if self.useSI:
            road_network = spatial_data_utils.load_neighbor(self.dataset, self.num_knn)
            for item in road_network:
                item = item.to(self.device)
        else:
            road_network = spatial_data_utils.load_netowrk(self.dataset).to(self.device)

        train_data_loader = spatial_data_utils.train_data_loader(self.config.spatial_train_set, self.train_batch)
        vali_data_loader,  vali_lenth = spatial_data_utils.vali_data_loader(
            self.config.spatial_vali_set, self.train_batch)

        best_epoch = 0
        best_hr10 = 0
        for epoch in range(0, self.epochs):
            net.train()
            losses = []
            start_train = time.time()
            for batch in tqdm(train_data_loader):
                optimizer.zero_grad()
                (data, coor, data_pos, coor_pos, data_neg, coor_neg,
                 data_pos_dis, data_neg_dis,
                 data_length, pos_length, neg_length) = batch
                a_embedding = net([road_network, distance_to_anchor_node], data, coor, data_length)
                p_embedding = net([road_network, distance_to_anchor_node], data_pos, coor_pos, pos_length)
                n_embedding = net([road_network, distance_to_anchor_node], data_neg, coor_neg, neg_length)

                loss = lossfunction(a_embedding, p_embedding, n_embedding, data_pos_dis, data_neg_dis, self.device)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            end_train = time.time()
            scheduler.step()
            logging.info(f'Used learning rate:{scheduler.get_last_lr()[0]}')
            logging.info('Epoch {}: Training time is {}, loss is {}'.format(
                epoch, (end_train - start_train), np.mean(losses)))
            if epoch % 1 == 0:
                net.eval()
                with torch.no_grad():
                    start_vali = time.time()
                    if self.useLSTM:
                        vali_embedding = torch.zeros((vali_lenth, self.hidden_size*2),
                                                     device=self.device, requires_grad=False)
                    else:
                        vali_embedding = torch.zeros((vali_lenth, self.hidden_size),
                                                     device=self.device, requires_grad=False)
                    vali_label = torch.zeros((vali_lenth, 50), requires_grad=False, dtype=torch.long)

                    for batch in tqdm(vali_data_loader):
                        (data, coor, label, data_length, idx) = batch
                        a_embedding = net([road_network, distance_to_anchor_node], data, coor, data_length)
                        vali_embedding[idx] = a_embedding
                        vali_label[idx] = label

                    acc = test_method.test_spa_model(vali_embedding, vali_label, self.device)
                    acc = acc.mean(axis=0)  # HR-10 HR-50 R10@50 R1@1 R1@10 R1@50
                    acc[0] = acc[0] / 10.0
                    acc[1] = acc[1] / 50.0
                    acc[2] = acc[2] / 10.0
                    logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                        self.dataset, self.distance_type, vali_lenth))
                    end_vali = time.time()
                    logging.info(f"vali time: {end_vali - start_vali}")
                    logging.info(acc)
                    logging.info(" ")
                    save_modelname = self.config.save_folder + "/epoch_%d.pt" % epoch
                    if not os.path.exists(self.config.save_folder):
                        os.makedirs(self.config.save_folder)
                    torch.save(net.state_dict(), save_modelname)

                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                        logging.info(f'best epoch: {best_epoch}')
                    if epoch - best_epoch >= self.early_stop:
                        logging.info(save_modelname)
                        break
# %===============================================================================================================================================


class GTrajST_Trainer(object):
    def __init__(self, config):

        self.feature_size = config.feature_size
        self.embedding_size = config.embedding_size
        self.date2vec_size = config.date2vec_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate
        self.concat = config.concat
        self.device = "cuda:" + str(config.cuda)
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs

        self.train_batch = config.gtraj["train_batch"]
        self.test_batch = config.gtraj["test_batch"]
        # self.traj_file = str(config.traj_file)
        # self.time_file = str(config.time_file)
        self.usePE = config.gtraj["usePE"]
        self.useSI = config.gtraj["useSI"]

        self.dataset = str(config.dataset)
        self.distance_type = str(config.distance_type)
        self.early_stop = config.early_stop
        self.config = config

    def ST_train(self, load_model=None, load_optimizer=None):
        logging.info('SPGMT on ' + self.dataset + ' with ' + self.distance_type)
        logging.info('positive num: ' + str(self.config.pos_num))
        logging.info('GPU: '+str(self.config.cuda))
        net = GraphTrajSTEncoder(feature_size=self.feature_size,
                                 embedding_size=self.embedding_size,
                                 date2vec_size=self.date2vec_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout_rate=self.dropout_rate,
                                 concat=self.concat,
                                 device=self.device,
                                 usePE=self.usePE,
                                 useSI=self.useSI,
                                 dataset=self.dataset).to(self.device)

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        milestones_list = [15, 30, 45, 60, 75, 90, 115]
        # milestones_list=[30, 60, 90, 120, 150]
        logging.info(milestones_list)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_list, gamma=0.2)

        lossfunction = SpaLossFun(self.train_batch, self.distance_type).to(self.device)

        distance_to_anchor_node = spatial_data_utils.load_region_node(self.dataset).to(self.device)
        road_network = spatial_data_utils.load_neighbor(self.dataset)
        for item in road_network:
            item = item.to(self.device)

        train_data_loader = spatial_data_utils.train_st_data_loader(self.config.spatial_train_set, self.train_batch)
        vali_data_loader,  vali_lenth = spatial_data_utils.vali_st_data_loader(
            self.config.spatial_vali_set, self.train_batch)

        best_epoch = 0
        best_hr10 = 0
        for epoch in range(0, self.epochs):
            net.train()
            losses = []
            start_train = time.time()
            for batch in tqdm(train_data_loader):
                optimizer.zero_grad()
                (data, coor, d2vec,
                 data_pos, coor_pos, d2vec_pos,
                 data_neg, coor_neg, d2vec_neg,
                 data_pos_dis, data_neg_dis,
                 data_length, pos_length, neg_length) = batch
                a_embedding = net([road_network, distance_to_anchor_node], data, coor, d2vec, data_length)
                p_embedding = net([road_network, distance_to_anchor_node], data_pos, coor_pos, d2vec_pos, pos_length)
                n_embedding = net([road_network, distance_to_anchor_node], data_neg, coor_neg, d2vec_neg, neg_length)

                loss = lossfunction(a_embedding, p_embedding, n_embedding, data_pos_dis, data_neg_dis, self.device)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            end_train = time.time()
            scheduler.step()
            logging.info(f'Used learning rate:{scheduler.get_last_lr()[0]}')
            logging.info('Epoch {}: Training time is {}, loss is {}'.format(
                epoch, (end_train - start_train), np.mean(losses)))
            if epoch % 1 == 0:
                net.eval()
                with torch.no_grad():
                    start_vali = time.time()
                    vali_embedding = torch.zeros((vali_lenth, 384), device=self.device, requires_grad=False)
                    vali_label = torch.zeros((vali_lenth, 50), requires_grad=False, dtype=torch.long)

                    for batch in tqdm(vali_data_loader):
                        (data, coor, d2vec, label, data_length, idx) = batch
                        a_embedding = net([road_network, distance_to_anchor_node], data, coor, d2vec, data_length)
                        vali_embedding[idx] = a_embedding
                        vali_label[idx] = label

                    acc = test_method.test_spa_model(vali_embedding, vali_label, self.device)
                    acc = acc.mean(axis=0)  # HR-10 HR-50 R10@50 R1@1 R1@10 R1@50
                    acc[0] = acc[0] / 10.0
                    acc[1] = acc[1] / 50.0
                    acc[2] = acc[2] / 10.0
                    logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                        self.dataset, self.distance_type, vali_lenth))
                    end_vali = time.time()
                    logging.info(f"vali time: {end_vali - start_vali}")
                    logging.info(acc)
                    logging.info(" ")
                    save_modelname = self.config.save_folder + "/epoch_%d.pt" % epoch
                    if not os.path.exists(self.config.save_folder):
                        os.makedirs(self.config.save_folder)
                    torch.save(net.state_dict(), save_modelname)

                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                        logging.info(f'best epoch: {best_epoch}')
                    if epoch - best_epoch >= self.early_stop:
                        logging.info(save_modelname)
                        break

    def ST_eval(self, load_model=None):
        logging.info('SPGMT on ' + self.dataset + ' with ' + self.distance_type)
        net = GraphTrajSTEncoder(feature_size=self.feature_size,
                                 embedding_size=self.embedding_size,
                                 date2vec_size=self.date2vec_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 dropout_rate=self.dropout_rate,
                                 concat=self.concat,
                                 device=self.device,
                                 usePE=self.usePE,
                                 useSI=self.useSI,
                                 dataset=self.dataset).to(self.device)
        if load_model != None:
            logging.info(load_model)
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)
            test_data_loader, test_lenth = spatial_data_utils.vali_st_data_loader(
                self.config.spatial_test_set, self.test_batch)

            logging.info(f'test size:{test_lenth}')

            road_network = spatial_data_utils.load_neighbor(self.dataset)
            for item in road_network:
                item = item.to(self.device)

            distance_to_anchor_node = spatial_data_utils.load_region_node(self.dataset).to(self.device)

            net.eval()
            with torch.no_grad():
                start_test_epoch = time.time()
                test_embedding = torch.zeros((test_lenth, 384), device=self.device, requires_grad=False)
                test_label = torch.zeros((test_lenth, 50), requires_grad=False, dtype=torch.long)

                for batch in tqdm(test_data_loader):
                    (data, coor, d2vec, label, data_length, idx) = batch
                    a_embedding = net([road_network, distance_to_anchor_node], data, coor, d2vec, data_length)
                    test_embedding[idx] = a_embedding
                    test_label[idx] = label
                end_test_epoch = time.time()
                acc = test_method.test_spa_model(test_embedding, test_label, self.device)
                acc = acc.mean(axis=0)  # HR-10 HR-50 R10@50 R1@1 R1@10 R1@50
                acc[0] = acc[0] / 10.0
                acc[1] = acc[1] / 50.0
                acc[2] = acc[2] / 10.0
                logging.info('Dataset: {}, Distance type: {}, f_num is {}'.format(
                    self.dataset, self.distance_type, test_lenth))
                end_test = time.time()
                logging.info(f"epoch test time: {end_test_epoch - start_test_epoch}")
                logging.info(f"all test time: {end_test - start_test_epoch}")
                logging.info(acc)
