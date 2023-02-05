# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8
"""
import os
from time import time
import argparse
from typing import List
import json
import tqdm
from datetime import datetime

import torch
from torchsummary import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import numpy as np

from sst.dataloader_audiofiles import DatasetAudioFiles
from sst.models.tcn import TCN
from sst.models.frontend import FrontEndAug, FrontEndNoAug
from sst.models.finetune import ClassModel, FeatureExtractor
from sst.utils.utils import get_model_filepaths, pack_model_data_paths
from sst.utils.tempo_utils import tempo_to_mirex, softmax_to_mirex
from sst.metrics import tempo_eval_basic_batch, accuracy1, accuracy2

TEMPO_RANGE = (0,300)
DEFAULT_CONFIG = './configs/eval.yaml'


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

def _split_and_batch(tensor: torch.Tensor, split_size_or_sections: int) -> torch.Tensor:
    '''Given a tensor, split it in parts of length split_size_or_sections and stack them in a batch tensor.'''
    # Check that tensor is at least as long as the length and pad if not
    if tensor.shape[-1] >= split_size_or_sections:
        tensor.squeeze_(dim=0) #squeeze batch dim that will be added again by the split operation
        splits = torch.split(tensor, split_size_or_sections, dim=-1)
        # drop last split if shorter
        if tensor.shape[-1]%split_size_or_sections != 0:
            splits = splits[:-1]
        # Pack splits into a batch tensor
        batch_tensor = torch.stack(splits, 0)
    else:
        print('Short input tensor of size:', tensor.shape)
        # Calculate pad length
        pad_len = split_size_or_sections - tensor.shape[-1]
        # Pad tensor so that it has exactly the length expected at input of model.
        batch_tensor = F.pad(tensor, (0,pad_len), "constant", 0) 
    return batch_tensor

def eval(model, frontend, data_loader, config, device):
    results = []
    for i, data in tqdm.tqdm(enumerate(data_loader)):
        # get the inputs; data is a list of [inputs, labels]
        input, tempo_true = data[0].to(device), data[1].to(device)

        inputs = _split_and_batch(input, config.dataset.model_input_num_samples)
        tempi  = tempo_true.repeat(inputs.shape[0], tempo_true.shape[1])
        mel_inputs, tempi, ts_rate = frontend(inputs, tempi)
        outputs = F.softmax(model(mel_inputs), dim=1)

        # tempo_pred = torch.median(F.relu(denormatise_tempo(outputs)), dim=0)[0]
        if config.eval.reduction=='mean':
            output_reduced = torch.mean(outputs, dim=0, keepdim=True)
        elif config.eval.reduction == 'median':
            output_reduced = torch.median(outputs, dim=0, keepdim=True)[0]
        else: 
            raise ValueError('unknown reduction method')

        labels_mirex = tempo_to_mirex(tempo_true)
        outputs_mirex = softmax_to_mirex(output_reduced, tempo_range=TEMPO_RANGE)
        tempo_pred = torch.Tensor([outputs_mirex[0][0]])        
        _, one_correct, _ = tempo_eval_basic_batch(labels_mirex, outputs_mirex)
        assert len(one_correct)==1

        track_results = {
            'tempo_pred': tempo_pred.item(),
            'tempo_true': tempo_true.item(),
            'one_correct': one_correct[0],
            'accuracy1': accuracy1(tempo_pred.item(), tempo_true.item(), tol=config.eval.tempo_tol),
            'accuracy2': accuracy2(tempo_pred.item(), tempo_true.item(), tol=config.eval.tempo_tol)
            }
        results.append(track_results)

    # Compute average stats
    results_summary = {
        'accuracy1': np.mean([x['accuracy1'] for x in results]),
        'accuracy2': np.mean([x['accuracy2'] for x in results]),
        'one_correct': np.mean([x['one_correct'] for x in results])
        }
    return results_summary, results


def save_results(results_summary: dict, results: List[dict], config: object):
    outdir = config.docker.outdir
    results_summary_filepath = os.path.join(outdir, 'results_summary.json')
    results_filepath = os.path.join(outdir, 'results.json')
    config_filepath = os.path.join(outdir, 'config.yaml')
    with open(results_summary_filepath, 'w') as f:
        json.dump(results_summary, f)
    with open(results_filepath, 'w') as f:
        json.dump(results, f)
    with open(config_filepath, 'w') as f:
        OmegaConf.save(config=config, f=f)
    

if __name__ == "__main__":
    # CLI argument parser 
    parser = argparse.ArgumentParser(description='Eval Script')

    parser.add_argument('--pretrained_model_filepath', metavar='pretrained_model_filepath', type=str, help='Path to pre-trained model file')
    parser.add_argument('--pretrained_config_filepath', metavar='pretrained_config_filepath', type=str, help='Path to pre-trained model config file')
    parser.add_argument('--config_file', metavar='config_file', type=str, help='Path to YAML config file')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', metavar='epochs', type=int, help='Number of Epochs')
    parser.add_argument('--num_workers', metavar='num_workers', type=int, help='Number of data loading workers')
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
    if not args.tensorboard_logdir == None:
        config.training.tensorboard_logdir = args.tensorboard_logdir

    # Make output dir if not already exist
    if not os.path.isdir(config.docker.outdir):
        os.mkdir('/opt/ml/model')
    # Make unique run_name based on timestamp
    config.run_name = config.config_basename + str(datetime.timestamp(datetime.now()))

    ##################### Set Dataset ######################
    dataset = DatasetAudioFiles(config.dataset)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.dataset.num_workers)

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

    feat_getter = FeatureExtractor(pre_trained_model, layers=['tempo_block.dense'])
    model = ClassModel(feat_getter, feature_layer='tempo_block.dense', input_units=16, output_units=300)

    model.load_state_dict(torch.load(model_data.model_filepath, map_location=device))
    model.eval()

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

    ################## Evaluating ###################
    start_time = time()
    print('Running evaluation...')
    results_summary, results = eval(model, frontend,  data_loader, config=config, device=device)

    print('results summary', results_summary)

    print('Saving Results...')
    save_results(results_summary, results, config)

    print('total time',time() - start_time)