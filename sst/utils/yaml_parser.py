# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

YAML parser
"""
import yaml

def yaml_loader(filepath):
    with open(filepath, 'r') as stream:
        data = yaml.safe_load(stream)
    return data

def yaml_writer(data, filepath):
    with open(filepath, 'w') as stream:
        yaml.safe_dump(data, stream)
