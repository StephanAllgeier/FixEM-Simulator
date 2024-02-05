import os
import tempfile

import mlflow
import numpy as np
import torch
from torch import optim

from GeneratingTraces_RCGAN.FEM import make_logger
from GeneratingTraces_RCGAN.FEM.dataimport import TimeSeriesFEM
from GeneratingTraces_RCGAN.FEM.models import RCGANGenerator, RCGANDiscriminator, RGANGenerator, RGANDiscriminator
from GeneratingTraces_RCGAN.FEM.samplers import SimpleSampler
from GeneratingTraces_RCGAN.FEM.trainers import SequenceTrainer

logger = make_logger(__file__)


def main(opt):
    if not os.path.exists(opt['savepath']):
        os.makedirs(opt['savepath'])
        print(f"Folder {opt['savepath']} has been created.")
    else:
        print(f"Folder {opt['savepath']} already exists.")
    logger.info(opt)
    batch_size = opt['batch_size'] if opt['batch_size'] != -1 else None

    dataset = TimeSeriesFEM(folderpath=opt['input_folder'],
                            transform=opt['dataset_transform'], vital_signs=opt['signals'], no_mean=opt['no_mean'],
                            slice_length=opt['slice_length'], input_freq=opt['input_freq'],
                            resample_freq=opt['resample_freq'])
    label_dist = dataset.label_dist
    X = torch.from_numpy(dataset.data).cuda()
    y = torch.from_numpy(
        dataset.labels).long().cuda()
    num_train, num_test, num_vali = (int(opt['split'][0] * X.shape[0]), int(opt['split'][1] * X.shape[0]),
                                     int(opt['split'][2] * X.shape[0]))
    rand_indices = np.arange(len(X))
    np.random.shuffle(rand_indices)
    train_indices, test_indices, vali_indices = (rand_indices[:num_train], rand_indices[num_train:num_train + num_test],
                                                 rand_indices[num_train + num_test:num_train + num_test + num_vali])
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    X_vali, y_vali = X[vali_indices], y[vali_indices]
    train_sampler = SimpleSampler(X_train, y_train, label_dist=label_dist, batch_size=batch_size)
    test_sampler = SimpleSampler(X_test, y_test, label_dist=label_dist, batch_size=batch_size)
    vali_sampler = SimpleSampler(X_vali, y_vali, label_dist=label_dist, batch_size=batch_size)

    num_classes = len(torch.unique(y.view(-1, y.size(-1))))
    RGAN = [RGANGenerator, RGANDiscriminator]
    RCGAN = [RCGANGenerator, RCGANDiscriminator]

    if opt['type'] == "RGAN":
        GANtype = RGAN
    elif opt['type'] == "RCGAN":
        GANtype = RCGAN
    network = {
        'generator': {
            'name': GANtype[0],
            'args': {
                'output_size': opt['signals'],
                'input_size': opt['noise_size'],
                'hidden_size': opt['hidden_size'],
                'num_layers': opt['num_layers'],
                'dropout': opt['gen_dropout'] if opt['num_layers'] != 1 else 0,
                'sequence_length': X.shape[1],
                'rnn_type': 'lstm',
                'label_size': 1,
                'noise_size': opt["noise_size"],
                'num_classes': num_classes if opt['type'] == 'RCGAN' else None,
                'label_embedding_size': opt['label_embedding_size'] if opt['type'] == 'RCGAN' else None
            },
            'optimizer': {
                'name': optim.Adam,
                'args': {
                    'lr': opt['lr']
                }
            }
        },
        'discriminator': {
            'name': GANtype[1],
            'args': {
                'input_size': opt['signals'],
                'hidden_size': opt['hidden_size'],
                'rnn_type': 'lstm',
                'sequence_length': X.shape[1],
                'num_classes': num_classes if opt['type'] == 'RCGAN' else None,
                'label_size': 1,
                'label_embedding_size': opt['label_embedding_size'] if opt['type'] == 'RCGAN' else None
            },
            'optimizer': {
                'name': optim.Adam,
                'args': {
                    'lr': opt['lr']
                }
            }
        }
    }

    logger.info(network)
    log_file_path = os.path.join(opt['savepath'], "LogNetwork.txt")
    with open(log_file_path, 'w') as file:
        file.write(str(network) + '\n' + str(opt))

    trainer = SequenceTrainer(models=network,
                              recon=None,
                              ncritic=opt['ncritic'],
                              epochs=opt['epochs'],
                              retain_checkpoints=1,
                              checkpoints=f"{MODEL_DIR}/",
                              mlflow_interval=opt['eval_interval'],
                              device=DEVICE,
                              noise_size=opt['noise_size'],
                              eval_frequency=1,
                              vali_set=X_vali,
                              savepath=opt['savepath'],
                              GANtype=opt['type'],
                              scale=opt['scale'],
                              resamp_frequency=opt['resample_freq'],
                              dataset_name=opt['dataset_name']
                              )

    if opt['type'] == 'RGAN':
        trainer.train_RGAN(dataloader=train_sampler)
    elif opt['type'] == 'RCGAN':
        trainer.train_RCGAN(dataloader=train_sampler)
    logger.info(trainer.generator)
    logger.info(trainer.discriminator)


if __name__ == '__main__':
    with mlflow.start_run():
        with tempfile.TemporaryDirectory() as MODEL_DIR:
            if torch.cuda.is_available():
                DEVICE = torch.device("cuda")
                torch.backends.cudnn.deterministic = True
            else:
                DEVICE = torch.device("cpu")
            logger.info(f'Running on device {DEVICE}')
            params_list = []

            for lr in [0.0003 + i * 0.0001 for i in range(3)]:
                for hidden_size in [500]:
                    params = {"lr": lr, "batch_size": 128, "hidden_size": hidden_size}
                    params_list.append(params)
            i = 0
            for params in params_list:
                opt = {
                    "lr": params["lr"],
                    "epochs": 2001,
                    "ncritic": 3,
                    "batch_size": params["batch_size"],
                    "dataset_transform": 'normalize',
                    "signals": 2,
                    "gen_dropout": 0.2,
                    "noise_size": 5,
                    "hidden_size": params["hidden_size"],
                    'num_layers': 1,
                    "flag": 'train',
                    "slice_length": 10,
                    "no_mean": True,
                    'type': 'RCGAN',
                    'savepath': fr"C:\\Users\\uvuik\\Desktop\\NewRuns12012024\\SameLR\\GazeBase_scale=0.2,f=100,len=5s\\RCGAN_Params_lr_{params['lr']}_bs_{params['batch_size']}_hs_{params['hidden_size']}_resample=100",
                    'split': [0.8, 0.1, 0.1],
                    'label_embedding_size': 1,
                    'input_folder': r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\GazeBase",
                    'eval_interval': 10,
                    'input_freq': 1000,
                    'resample_freq': 100,
                    'scale': 0.2,
                    'dataset_name': 'GazeBase'
                }
                i += 1
                main(opt)
                print(f"{i / len(params_list) * 100}% done...")
