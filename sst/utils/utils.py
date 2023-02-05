# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

General utils
"""
import os
import tarfile
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple
import glob

from torch.utils.tensorboard import SummaryWriter

def download_from_s3(s3_path, local_path):
    cmd = 'aws s3 cp {} {}'.format(s3_path, local_path)
    os.system(cmd)
    local_filepath = os.path.join(local_path, os.path.basename(s3_path))
    return local_filepath

def expand_tar(local_tar_filepath, expand_dir):
    '''Expand an local tar file and return the folder name'''
    @dataclass
    class ModelData:
        model_filepath: str
        config_filepath: str
        basename: str

    with tarfile.open(local_tar_filepath, mode='r:*') as f:
        f.extractall(path=expand_dir)
        for member in f.getmembers():
            name = member.name
            if name.endswith('.pt'):
                model_filepath = os.path.join(expand_dir, name)
            elif name.endswith('.yaml'):
                config_filepath = os.path.join(expand_dir, name)
            else:
                raise ValueError('unknown file extension in archive')

    basename = os.path.splitext(os.path.basename(model_filepath))[0]
    model_data = ModelData(
        basename=basename,
        model_filepath=model_filepath,
        config_filepath=config_filepath
        )
    return model_data

def pack_model_data_paths(model_filepath: str, config_filepath: str):
    '''Pack model and config local filepaths into a model_data object for convenience'''
    @dataclass
    class ModelData:
        model_filepath: str
        config_filepath: str
        basename: str
    
    basename = os.path.splitext(os.path.basename(model_filepath))[0]
    model_data = ModelData(
        basename=basename,
        model_filepath=model_filepath,
        config_filepath=config_filepath
        )
    return model_data


def download_and_expand_model(s3_model_path, local_dir):
    '''Download pre-trained model from S3 to local_dir and expand the archive'''
    local_archive = download_from_s3(s3_model_path, local_dir)
    archive_dir = os.path.join(local_dir, 'model')
    model_data = expand_tar(local_archive, archive_dir)
    return model_data

def set_tensorboard_writer(tensorboard_logdir, run_name=None):
    if tensorboard_logdir == None:
        # If no directory is provided, use Pytorch's default location.
        writer = SummaryWriter(log_dir=tensorboard_logdir)
    else:
        if run_name==None:
            run_name = str(datetime.timestamp(datetime.now()))
        writer = SummaryWriter(log_dir=os.path.join(tensorboard_logdir, run_name))
    return writer

def get_model_filepaths(model_dir: str) -> Tuple[str, str]: 
    '''Get the pre-trained model and config file paths, given the model directory.
    This script expects that the model directory contains: exactly one model file with .pt extension and exactly one config with .yaml extension.'''
    
    model_filepaths = glob.glob(os.path.join(model_dir,"*.pt"))
    config_filepaths = glob.glob(os.path.join(model_dir,"*.yaml"))
    
    # Look for model file
    if len(model_filepaths)>1:
        raise ValueError('Found multiple model files (with .pt extension). Pre-trained model folder should contain exactly 1 such file.')
    elif len(model_filepaths)<1:
        raise ValueError('Cannot find a pretrained model file. Model file should have the extension .pt and be placed in the pretrained_model folder.')
    else:
        model_filepath = model_filepaths[0]
    
    # Look for config file
    if len(config_filepaths)>1:
        raise ValueError('Found multiple model config files (with .yaml extension). Pre-trained model folder should contain exactly 1 such file.')
    elif len(config_filepaths)<1:
        raise ValueError('Cannot find a pretrained model config file. Model file should have the extension .yaml and be placed in the pretrained_model folder.')
    else:
        config_filepath = config_filepaths[0]
    
    return model_filepath, config_filepath
