# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8
"""
import os
from time import time
from datetime import datetime
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from omegaconf import OmegaConf

from sst.dataloader_audiofiles import DatasetAudioFiles
from sst.models.tcn import TCN
from sst.models.frontend import FrontEndAug, FrontEndNoAug
from sst.models.finetune import ClassModel, FeatureExtractor
from sst.utils.utils import set_tensorboard_writer, pack_model_data_paths, get_model_filepaths
from sst.utils.tempo_utils import softmax_to_mirex, tempo_to_mirex, tempo_to_class_index
from sst.metrics import tempo_eval_basic_batch
from sst.losses.crossentropy import XentBoeck


TEMPO_RANGE = (0,300)
DEFAULT_CONFIG = './configs/finetune.yaml'


def freeze_model(model, freeze_output_layer=False):
    '''Freeze network parameters. Optionally keep the output layer trainable (to transfer to new data).'''
    for parameter in model.parameters():
        parameter.requires_grad = False
    if not freeze_output_layer:
        for param in model.tempo_block.output_reg.parameters():
            param.requires_grad = True
    return model

def freeze_pre_trained_model(model):
    '''Freeze pre-trained network parameters. Optionally keep the output layer trainable (to transfer to new data).'''
    # freeze all parameters of pre-trained model.
    for parameter in model.pre_trained_model.parameters():
        parameter.requires_grad = False
    # unfreeze parameters of the fresh output layer.
    for param in model.output.parameters():
        param.requires_grad = True
    return model

def get_tensorboard_hparams(config):
    hparams = {"loss": config.training.loss,
               "batch_size": config.training.batch_size,
               "opt": config.training.opt.opt_name,
               "lr": config.training.lr,
               }
    return hparams

def save_model_files(model, config):
    '''Saves the trained model and the corresponding YAML config file.'''
    ############## Save Model ##################
    model_path = os.path.join(config.docker.outdir, '{}.pt'.format(config.run_name))
    torch.save(model.state_dict(), model_path)
    ############## Save config ##################
    config_path = os.path.join(config.docker.outdir, '{}_config.yaml'.format(config.run_name))
    OmegaConf.save(config=config, f=config_path)


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
        running_one_correct = []
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, tempi = data[0].to(device), data[1].to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            mel_inputs, tempi, _ = frontend(inputs, tempi)
            outputs = model(mel_inputs)

            labels_cls_idx = tempo_to_class_index(tempi, TEMPO_RANGE)

            labels_mirex = tempo_to_mirex(tempi)
            outputs_mirex = softmax_to_mirex(F.softmax(outputs, dim=1))
            _, one_correct, _ = tempo_eval_basic_batch(labels_mirex, outputs_mirex)
            running_one_correct.extend(one_correct)

            loss = criterion(outputs, labels_cls_idx.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

            writer.add_scalar('Loss_batch/train', loss.item(), n_batches)
            n_batches += 1
            running_loss += loss.item()

        avg_train_loss = running_loss / (i + 1)
        avg_one_correct = np.mean(running_one_correct)
        writer.add_scalar('Loss_epoch/train', avg_train_loss, epoch)
        writer.add_scalar('One_correct_epoch/train', avg_one_correct, epoch)

        ###### Validation loop #####
        if val_loader == None:
            avg_val_loss = -1
            avg_val_one_correct = -1
        else:
            with torch.no_grad():
                total_val_loss = 0
                one_correct_val = []
                for i, data in enumerate(val_loader):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, tempi = data[0].to(device), data[1].to(device)

                    mel_inputs, tempi, ts_rate = frontend(inputs, tempi)
                    outputs = model(mel_inputs)

                    labels_cls_idx = tempo_to_class_index(tempi, TEMPO_RANGE)

                    loss = criterion(outputs, labels_cls_idx.to(device))
                    total_val_loss += loss.item()
                    
                    labels_mirex = tempo_to_mirex(tempi)
                    outputs_mirex = softmax_to_mirex(F.softmax(outputs, dim=1))
                    _, one_correct, _ = tempo_eval_basic_batch(labels_mirex, outputs_mirex)
                    one_correct_val.extend(one_correct)
                    # print statistics
                    print('[%d, %5d] Val loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                avg_val_loss = total_val_loss / (i + 1)
                avg_val_one_correct = np.mean(one_correct_val)

                print('[%d, %5d] Avg val loss: %.3f' % (epoch + 1, i + 1, avg_val_loss))
                writer.add_scalar('Loss_epoch/val', avg_val_loss, epoch)
                writer.add_scalar('One_correct_epoch/val', avg_val_one_correct, epoch)


    # Record hparams and final metrics
    writer.flush()
    tensorboard_hparams = get_tensorboard_hparams(config)
    writer.add_hparams(
        tensorboard_hparams, 
        {
            'hparam/train_loss': avg_train_loss, 
            'hparam/val_loss': avg_val_loss,
            'hparam/train_onecorrect': avg_one_correct,
            'hparam/val_onecorrect': avg_val_one_correct
        },
        run_name='hparams'
        )
    writer.flush()
    writer.close()
    return model



if __name__ == "__main__":

    # CLI argument parser
    parser = argparse.ArgumentParser(description='Finetune Script')

    parser.add_argument('--config_file', metavar='config_file', type=str, help='Path to finetune config file')
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
    dataset = DatasetAudioFiles(config.dataset)

    trainset, valset = random_split(
        dataset,
        [round(config.dataset.splits.train*len(dataset)), round(config.dataset.splits.val*len(dataset))],
        generator=torch.Generator().manual_seed(config.dataset.splits.rseed)
        )
    
    train_loader = DataLoader(
        trainset, 
        batch_size=config.training.batch_size, 
        shuffle=config.training.shuffle,
        num_workers=config.training.num_workers
        )
    
    val_loader = DataLoader(
        valset, 
        batch_size=config.training.batch_size, 
        shuffle=config.training.shuffle,
        num_workers=config.training.num_workers
        )

    ##################### Set Device ###################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE", device, flush=True)

    ##################### Set model ######################
    print('Loading pre-trained model config...')
    pretrained_model_filepath, pretrained_config_filepath = get_model_filepaths(config.pretrained_model_dir)
    model_data = pack_model_data_paths(pretrained_model_filepath, pretrained_config_filepath)
    config_pretrain = OmegaConf.load(model_data.config_filepath)
    config = OmegaConf.merge(config_pretrain, config)
    print('Loading pre-trained model...')
    pre_trained_model = TCN(
        num_filters=config.model.num_filters,
        tempo_range=TEMPO_RANGE, 
        mode=config.model.mode,
        dropout_rate=config.model.dropout,
        add_proj_head=config.model.add_proj_head, 
        proj_head_dim=config.model.proj_head_dim
        )
    pre_trained_model.load_state_dict(torch.load(model_data.model_filepath, map_location=device))
    # Set forward hook to get output of the dense layer before regression (1 unit) output. 
    feat_getter = FeatureExtractor(pre_trained_model, layers=['tempo_block.dense'])
    model = ClassModel(feat_getter, feature_layer='tempo_block.dense', input_units=16, output_units=300)

    summary(model, (1, 81,1361), depth=3)
    model.to(device)
    print("model", model)


    ############# Set front-end pre-processing ###########
    if config.frontend.use_augmentations:
        print('Frontend Augmentations ON')
        frontend = FrontEndAug(config.frontend)
    else:
        print('Frontend Augmentations OFF')
        frontend = FrontEndNoAug(config.frontend)
    frontend.to(device)

    ############### Set Training params ################
    print('Setting Training Parameters...')

    if config.training.loss=='CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif config.training.loss=='Boeck':
        criterion = XentBoeck(device=device)
    else:
        raise ValueError('Unknown loss function')

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