# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8
"""
import os
from time import time
from datetime import datetime
import argparse
import json

import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from dataloader_audiofiles import DatasetDualAug
from models.tcn import TCN
from sst.models.frontend import FrontEndAug, FrontEndNoAug
from sst.losses.l1 import L1Ratio, L1Diff

TEMPO_RANGE = (0,300)
DEFAULT_CONFIG = './configs/train.yaml'


def save_model_files(model, config):
    '''Saves the trained model and the corresponding YAML config file.'''
    ############## Save Model ##################
    model_path = os.path.join(config.docker.outdir, '{}.pt'.format(config.run_name))
    torch.save(model.state_dict(), model_path)
    ############## Save config ##################
    config_path = os.path.join(config.docker.outdir, '{}_config.yaml'.format(config.run_name))
    OmegaConf.save(config=config, f=config_path)

def checkpoint_model_files(model, config, checkpoint_basename):
    '''Saves the trained model and the corresponding YAML config file.'''
    ############## Save Model ##################
    model_path = os.path.join(config.docker.checkpoint_dir, '{}_{}.pt'.format(config.run_name, checkpoint_basename))
    torch.save(model.state_dict(), model_path)
    ############## Save config ##################
    config_path = os.path.join(config.docker.checkpoint_dir, '{}_config.yaml'.format(config.run_name))
    OmegaConf.save(config=config, f=config_path)


def set_tensorboard_writer(tensorboard_logdir, run_name=None):
    if tensorboard_logdir == None:
        # If no directory is provided, use Pytorch's default location.
        writer = SummaryWriter(log_dir=tensorboard_logdir)
    else:
        if run_name==None:
            run_name = str(datetime.timestamp(datetime.now()))
        writer = SummaryWriter(log_dir=os.path.join(tensorboard_logdir, run_name))
    return writer

def get_tensorboard_hparams(config):
    hparams = {
        "num_filters": config.model.num_filters,
        "loss": config.training.loss,
        "batch_size": config.training.batch_size,
        "opt": config.training.opt.opt_name,
        "lr": config.training.lr,
        "dropout": config.model.dropout,
        "use_augmentations": config.frontend.use_augmentations,
        "aug_timestretch": 'timestretch' in config.frontend.augmentations,
        "aug_volume": 'volume' in config.frontend.augmentations,
        "aug_pol_inv": 'polarity_inversion' in config.frontend.augmentations,
        "aug_gaussnoise": 'gaussian_noise' in config.frontend.augmentations,
        "aug_freq_masking": 'freq_masking' in config.frontend.augmentations,
        }
    return hparams

def validate_device(device):
    if device == None:
        raise ValueError("device argument not provided. Please provide a device")

def train(model, frontend, criterion, optimizer, train_loader, config, val_loader=None, device=None):
    ################## Validate device #################
    validate_device(device)
    ################## Set Tensorboard #################
    writer = set_tensorboard_writer(config.training.tensorboard_logdir, run_name=config.run_name)
    ################## Training loop ###################
    print('Training...')
    n_batches = 0
    for epoch in range(config.training.epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs_i, tempi_i, ts_rates_i = data[0][0].to(device), data[0][1].to(device), data[0][2].to(device)
            inputs_j, tempi_j, ts_rates_j = data[1][0].to(device), data[1][1].to(device), data[1][2].to(device)
            ts_rates_i = torch.unsqueeze(ts_rates_i,1)
            ts_rates_j = torch.unsqueeze(ts_rates_j,1)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            melspecs_i, _, _ = frontend(inputs_i, tempi_i)
            melspecs_j, _, _ = frontend(inputs_j, tempi_j)

            z_i = model(melspecs_i)
            z_j = model(melspecs_j)

            tempo_loss = 0.5*(criterion(z_i, z_j, ts_rates_i, ts_rates_j) + criterion(z_j, z_i, ts_rates_j, ts_rates_i))
            loss = tempo_loss

            loss.backward()

            optimizer.step()
            # print statistics
            z_mag_mean = 0.5 * (torch.mean(torch.abs(z_i)) + torch.mean(torch.abs(z_j)))
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))
            writer.add_scalar('z_metrics_train/z_mag_mean', z_mag_mean.item(), n_batches)
            writer.add_scalar('z_metrics_train/tempo_loss', tempo_loss.item(), n_batches)
            writer.add_scalar('Loss_batch/train', loss.item(), n_batches)
            n_batches += 1

            running_loss += loss.item()

        avg_train_loss = running_loss / (i + 1)
        writer.add_scalar('Loss_epoch/train', avg_train_loss, epoch)

        ###### Validation loop #####
        if val_loader == None:
            avg_val_loss = -1
        else:
            with torch.no_grad():
                total_val_loss = 0
                total_val_tempo_loss = 0
                for i, data in enumerate(val_loader):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs_i, tempi_i, ts_rates_i = data[0][0].to(device), data[0][1].to(device), data[0][2].to(device)
                    inputs_j, tempi_j, ts_rates_j = data[1][0].to(device), data[1][1].to(device), data[1][2].to(device)
                    ts_rates_i = torch.unsqueeze(ts_rates_i, 1)
                    ts_rates_j = torch.unsqueeze(ts_rates_j, 1)

                    melspecs_i, _, _ = frontend(inputs_i, tempi_i)
                    melspecs_j, _, _ = frontend(inputs_j, tempi_j)

                    z_i = model(melspecs_i)
                    z_j = model(melspecs_j)

                    tempo_loss = 0.5*(criterion(z_i, z_j, ts_rates_i, ts_rates_j) + criterion(z_j, z_i, ts_rates_j, ts_rates_i))
                    loss = tempo_loss

                    total_val_loss += loss.item()
                    total_val_tempo_loss += tempo_loss.item()

                    # print statistics
                    print('[%d, %5d] Val loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                avg_val_loss = total_val_loss / (i + 1)
                avg_val_tempo_loss = total_val_tempo_loss / (i+1)

                print('[%d, %5d] Avg val loss: %.3f' % (epoch + 1, i + 1, avg_val_loss))
                writer.add_scalar('Loss_epoch/val', avg_val_loss, epoch)
                writer.add_scalar('Loss_epoch/val_tempo_loss', avg_val_tempo_loss, epoch)

        ####### Checkpoint model #######
        if config.training.checkpoint:
            checkpoint_basename = 'chekpoint_epoch_{}'.format(epoch)
            checkpoint_model_files(model, config, checkpoint_basename)

    # Record hparams and final metrics
    writer.flush()
    tensorboard_hparams = get_tensorboard_hparams(config)
    writer.add_hparams(tensorboard_hparams, {'hparam/train_loss': avg_train_loss, 'hparam/val_loss': avg_val_loss},
                       run_name='hparams')
    writer.flush()
    writer.close()
    return model



if __name__ == "__main__":

    # CLI argument parser
    parser = argparse.ArgumentParser(description='Training Script')

    parser.add_argument('--config_file', metavar='config_file', type=str, help='Path to YAML config file')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', metavar='epochs', type=int, help='Number of Epochs')
    parser.add_argument('--loss', metavar='loss', type=str, help='Loss')
    parser.add_argument('--lr', metavar='lr', type=float, help='Learning Rate')
    parser.add_argument('--num_workers', metavar='num_workers', type=int, help='Number of data loading workers')
    parser.add_argument('--opt', metavar='opt', type=str, help='Optimizer name')
    parser.add_argument('--tensorboard_logdir', metavar='tensorboard_logdir', type=str, help='Tensorboard logdir')

    args = parser.parse_args()
    print('args', args)

    if args.config_file==None:
        config = OmegaConf.load(DEFAULT_CONFIG)
    else:
        conf_base = OmegaConf.load(DEFAULT_CONFIG)
        conf_user = OmegaConf.load(args.config_file)
        config = OmegaConf.merge(conf_base, conf_user)

    if not args.batch_size == None:
        config.training.batch_size = args.batch_size
    if not args.epochs == None:
        config.training.epochs = args.epochs
    if not args.loss == None:
        config.training.loss = args.loss
    if not args.lr == None:
        config.training.lr = args.lr
    if not args.opt == None:
        config.training.opt.opt_name = args.opt
    if not args.tensorboard_logdir == None:
        config.training.tensorboard_logdir = args.tensorboard_logdir


    # Make output dir if not already exist
    if not os.path.isdir(config.docker.outdir):
        os.mkdir('/opt/ml/model')
    # Make unique run_name based on timestamp
    config.run_name = config.config_basename + str(datetime.timestamp(datetime.now()))

    ##################### Set Dataset ######################
    dataset = DatasetDualAug(config.dataset)
    
    trainset, valset = random_split(
        dataset,
        [round(config.dataset.splits.train*len(dataset)), round(config.dataset.splits.val*len(dataset))],
        generator=torch.Generator().manual_seed(config.dataset.splits.rseed)
        )
    
    train_loader = DataLoader(
        trainset, 
        batch_size=config.training.batch_size, 
        shuffle=config.training.shuffle,
        num_workers=config.training.num_workers, 
        drop_last=True
        )
    
    val_loader = DataLoader(
        valset, 
        batch_size=config.training.batch_size, 
        shuffle=config.training.shuffle,
        num_workers=config.training.num_workers, 
        drop_last=True
        )


    ##################### Set Device ###################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE", device, flush=True)

    ############# Set front-end pre-processing ###########
    if config.frontend.use_augmentations:
        frontend = FrontEndAug(config.frontend)
    else:
        frontend = FrontEndNoAug(config.frontend)
    frontend.to(device)

    ##################### Set model ######################
    model = TCN(
        num_filters=config.model.num_filters,
        tempo_range=TEMPO_RANGE, 
        mode=config.model.mode,
        dropout_rate=config.model.dropout,
        add_proj_head=config.model.add_proj_head, 
        proj_head_dim=config.model.proj_head_dim
        )
    summary(model, (1, 81,1361), depth=3)
    model.to(device)
    print("model", model)


    ############### Set Training params ################
    print('Setting Training Parameters...')
    if config.training.loss=='l1_ratio':
        criterion = L1Ratio(reduction='mean')
    elif config.training.loss=='l1_diff':
        criterion = L1Diff(reduction='mean')

    if config.training.opt.opt_name=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.training.lr)
    elif config.training.opt.opt_name=='adam':
        optimizer = optim.Adam(model.parameters(), lr=config.training.lr, weight_decay=0)
    elif config.training.opt.opt_name=='adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=config.training.lr, rho=0.9, eps=1e-06, weight_decay=0)


    ################## Training ###################
    start_time = time()

    model = train(
        model, 
        frontend,
        criterion,
        optimizer, 
        train_loader=train_loader, 
        config=config,
        val_loader=val_loader, 
        device=device
        )

    print('Saving Model files...')
    save_model_files(model, config)

    print('total time',time() - start_time)