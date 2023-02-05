# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Classes for fine-tuning pre-trained models (e.g. by adding a linear layer on top).
"""
from typing import Dict, Iterable, Callable

import torch
import torch.nn as nn

EPS = 10e-8


class LinTransformModel(nn.Module):
    '''Add a linear transformation to the output of pre-trained model'''
    def __init__(self, pre_trained_model: nn.Module,
                 lin_input_units: int = 1, lin_output_units: int = 1,
                 freeze_pre_trained: bool = False):
        super(LinTransformModel, self).__init__()
        self.pre_trained_model = pre_trained_model
        self.freeze_pre_trained = freeze_pre_trained
        if self.freeze_pre_trained is True:
            self.pre_trained_model.eval()
        self.lin_input_units = lin_input_units
        self.lin_output_units = lin_output_units
        # add a linear output layer
        self.output = nn.Linear(self.lin_input_units, self.lin_output_units)

    def train(self):
        super().train()
        if self.freeze_pre_trained is True:
            self.pre_trained_model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_trained_model(x)
        x = self.output(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        # print('FeatureExtractor named module keys:', dict([*self.model.named_modules()]).keys())
        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
        return self._features

class ClassificationHead(nn.Module):
    '''Add a classification head after the last dense (linear) layer of the model.'''
    def __init__(self, 
                input_units: int = 16,
                output_units: int = 300):
        super(ClassificationHead, self).__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.activation = nn.ELU()
        self.output = nn.Linear(self.input_units, self.output_units)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.output(x)
        return x

class ClassModel(nn.Module):
    '''Add a classification layer after a feature extractor (e.g. intermediate layer of pre-trained model)'''
    def __init__(self, feature_extractor: nn.Module,
                        feature_layer: str = 'tempo_block.dense',
                        input_units: int = 16,
                        output_units: int = 300, 
                        freeze_feat_extractor: bool = True):
        super(ClassModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_layer = feature_layer
        self.input_units = input_units
        self.output_units = output_units
        self.class_head = ClassificationHead(input_units=self.input_units, output_units=self.output_units)
        self.freeze_feat_extractor = freeze_feat_extractor
        if self.freeze_feat_extractor==True:
            self.feature_extractor.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)[self.feature_layer]
        x = self.class_head(x)
        return x




