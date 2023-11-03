import os

import mlflow
import tempfile
import click
import torch
import numpy as np
from GeneratingTraces_RGANtorch.FEM.logs import log_model, log_df
from GeneratingTraces_RGANtorch.FEM.models import RCGANGenerator, RCGANDiscriminator, RGANGenerator, RGANDiscriminator
from GeneratingTraces_RGANtorch.FEM.utils import synthesis_df, numpy_to_cuda, tile
from GeneratingTraces_RGANtorch.FEM.metrics import mean_feature_error, classify, tstr
from GeneratingTraces_RGANtorch.ward2icu.samplers import BinaryBalancedSampler, SimpleSampler

from GeneratingTraces_RGANtorch.FEM.trainers import SequenceTrainer
from torch import optim
from torchgan import losses
from GeneratingTraces_RGANtorch.FEM import make_logger
from GeneratingTraces_RGANtorch.FEM.dataimport import TimeSeriesFEM
from sklearn.model_selection import train_test_split
logger = make_logger(__file__)
from torch.utils.data import DataLoader, random_split

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def main():
    opt = {
        "lr": 0.00005,
        "epochs": 5000,
        "ncritic": 3,
        "batch_size": 32,
        "dataset_transform": 'normalize',
        "signals": 2,
        "gen_dropout": 0.2,
        "noise_size": 20,
        "hidden_size": 100,
        'num_layers': 2,
        "flag": 'train',
        "slice_length": 2,
        "no_mean": True,
        'type': 'RCGAN',
        'savepath': r"C:\Users\uvuik\Desktop\Torch\RCGAN_Test4",
        'split': [0.8, 0.1, 0.1],
        'label_embedding_size': 5,
        'input_folder': r"C:\Users\uvuik\bwSyncShare\Documents\Dataset\TrainingData\Roorda",
        'eval_interval': 20

    }
    if not os.path.exists(opt['savepath']):
        os.makedirs(opt['savepath'])
        print(f"Der Ordner {opt['savepath']} wurde erstellt.")
    else:
        print(f"Der Ordner {opt['savepath']} existiert bereits.")
    logger.info(opt)
    batch_size = opt['batch_size'] if opt['batch_size'] != -1 else None

    # dataset = TimeSeriesVitalSigns(transform=opt['dataset_transform'],
    #                               vital_signs=opt['signals'])

    dataset = TimeSeriesFEM(folderpath=opt['input_folder'],
                            transform=opt['dataset_transform'], vital_signs=opt['signals'], no_mean=opt['no_mean'],
                            slice_length=opt['slice_length'], input_freq=1920, resample_freq=250)
    X = torch.from_numpy(dataset.data).cuda()
    y = torch.from_numpy(
        dataset.labels).long().cuda()#TODO Das ganze soll nur ein 714,500 tensor sein, nicht noch ,1 als tensorfunktion, .squeeze() entfernen
    num_train, num_test, num_vali = int(opt['split'][0] * X.shape[0]), int(opt['split'][1] * X.shape[0]), int(opt['split'][2] * X.shape[0])
    rand_indices = np.arange(len(X))
    np.random.shuffle(rand_indices)
    train_indices, test_indices, vali_indices = (rand_indices[:num_train], rand_indices[num_train:num_train + num_test],
                                                 rand_indices[num_train + num_test:num_train + num_test + num_vali])
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    X_vali, y_vali = X[vali_indices], y[vali_indices]
    train_sampler = SimpleSampler(X_train, y_train, batch_size=batch_size)
    test_sampler = SimpleSampler(X_test, y_test, batch_size=batch_size)
    vali_sampler = SimpleSampler(X_vali, y_vali, batch_size=batch_size)


    num_classes = len(torch.unique(y.view(-1, y.size(-1))))
    RGAN = [RGANGenerator, RGANDiscriminator]
    RCGAN = [RCGANGenerator, RCGANDiscriminator]
    # TODO: Anpassen
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
                'noise_size': opt["noise_size"],
                'num_classes': num_classes if opt['type'] == 'RCGAN' else None,
                'label_embedding_size': opt['label_embedding_size'] if opt['type'] == 'RCGAN' else None
            },
            'optimizer': {
                'name': optim.Adam,  # optim.RMSprop,
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
                'label_embedding_size': opt['label_embedding_size'] if opt['type'] == 'RCGAN' else None
            },
            'optimizer': {
                'name': optim.Adam,  # optim.RMSprop, originalPaper nimmt GradientDescend
                'args': {
                    'lr': opt['lr']
                }
            }
        }
    }

    wasserstein_losses = [losses.WassersteinGeneratorLoss(),
                          losses.WassersteinDiscriminatorLoss(),
                          losses.WassersteinGradientPenalty()]

    logger.info(network)
    with open(f"{opt['savepath']}/LogNetwork.txt", 'w') as file:
        file.write(str(network) + '\n' + str(opt))

    trainer = SequenceTrainer(models=network,
                              recon=None,
                              ncritic=opt['ncritic'],
                              losses_list=wasserstein_losses,
                              epochs=opt['epochs'],
                              retain_checkpoints=1,
                              checkpoints=f"{MODEL_DIR}/",
                              mlflow_interval=opt['eval_interval'],
                              device=DEVICE,
                              noise_size=opt['noise_size'],
                              eval_frequency=1,
                              vali_set=X_vali,
                              savepath=opt['savepath']
                              )

    if opt['type'] == 'RGAN':
        trainer.train_RGAN(dataloader=train_sampler)
    elif opt['type'] == 'RCGAN':
        trainer.train_RCGAN(dataloader=train_sampler)
    logger.info(trainer.generator)
    logger.info(trainer.discriminator)
    #Generator = trainer.generator, Diskriminator = trainer.diskriminator

    df_synth, X_synth, y_synth = synthesis_df(trainer.generator, dataset)

    logger.info(df_synth.sample(10))
    logger.info(df_synth.groupby('cat_vital_sign')['value'].nunique()
                .div(df_synth.groupby('cat_vital_sign').size()))
    X_real = X.detach().cpu().numpy()
    mfe = np.abs(mean_feature_error(X_real, X_synth))
    logger.info(f'Mean feature error: {mfe}')

    mlflow.set_tag('flag', opt['flag'])
    log_df(df_synth, 'synthetic/vital_signs')
    mlflow.log_metric('mean_feature_error', mfe)

    trainer_class = classify(X_synth, y_synth, epochs=2_000, batch_size=batch_size)
    trainer_tstr = tstr(X_synth, y_synth, X, y, epochs=3_000, batch_size=batch_size)
    log_model(trainer_class.model, 'models/classifier')
    log_model(trainer_tstr.model, 'models/tstr')


if __name__ == '__main__':
    with mlflow.start_run():
        with tempfile.TemporaryDirectory() as MODEL_DIR:
            if torch.cuda.is_available():
                DEVICE = torch.device("cuda")
                torch.backends.cudnn.deterministic = True
            else:
                DEVICE = torch.device("cpu")
            logger.info(f'Running on device {DEVICE}')
            main()