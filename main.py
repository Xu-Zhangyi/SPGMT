from Trainer import GTrajST_Trainer, GTrajsim_Trainer
from setting import SetParameter, setup_logger, zipdir
import datetime
import os.path as osp
import os
import zipfile
import pathlib
import logging
config = SetParameter()
if __name__ == '__main__':
    # %====================================================================

    # -----------train set-----------
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(osp.join('log1', config.dataset)):
        os.makedirs(osp.join('log1', config.dataset))
    config.save_folder = osp.join('saved_models', config.dataset, config.distance_type + '_' + str(current_time))
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)
    setup_logger(osp.join('log', config.dataset, config.distance_type + '_' + str(current_time) + '_train.log'))
    zipf = zipfile.ZipFile(os.path.join(config.save_folder, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    # -----------train SPGMT-----------
    GTrajSim = GTrajsim_Trainer(config)
    logging.info(f'weights for different neighbors (alpha1,alpha2):({config.alpha1},{config.alpha2})')
    logging.info(f'number of nearest neighbors:{config.num_knn}')
    logging.info(f'hidden size:{config.hidden_size}')
    GTrajSim.Spa_train()

    # %====================================================================

    # # -----------test set-----------
    # current_time = '20241101_125365'
    # load_model_name = osp.join('saved_models', config.dataset, config.distance_type +
    #                            '_' + current_time, 'epoch_120.pt')
    # setup_logger(osp.join('log', config.dataset, config.distance_type + '_' + str(current_time) + '_test.log'))
    # # -----------test SPGMT-----------
    # GTrajSim = GTrajsim_Trainer(config)
    # logging.info(f'weights for different neighbors (alpha1,alpha2):({config.alpha1},{config.alpha2})')
    # logging.info(f'number of nearest neighbors:{config.num_knn}')
    # logging.info(f'hidden size:{config.hidden_size}')
    # GTrajSim.Spa_eval(load_model=load_model_name, is_long_traj=False)

    # %====================================================================

    # # -----------train SPGMT(spatio-temporal)-----------
    # GTrajST = GTrajST_Trainer(config)
    # GTrajST.ST_train(load_model=load_model_name, load_optimizer=load_optimizer_name)
    # # -----------test SPGMT(spatio-temporal)-----------
    # current_time = '20241112_070611'
    # load_model_name = osp.join('saved_models', config.dataset, config.distance_type
    #                            + '_' + current_time,'epoch_12.pt')
    # setup_logger(osp.join('log', config.dataset,
    #              config.distance_type + '_' + str(current_time) + '_test.log'))
    # GTrajST.ST_eval(load_model=load_model_name)
