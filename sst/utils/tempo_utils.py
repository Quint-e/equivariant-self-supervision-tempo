# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Tempo utils functions

"""
import numpy as np
import torch
import torchaudio as ta
import torch.nn.functional as F


def tempo_to_onehot(tempo, tempo_range, tempo_widening_win=None):
    '''Converts float tempo values into a one-hot vector with tempi linearly distributed within the tempo_range (tempo 0 indicates no tempo).
    Optionally widens the one-hot encoding by convolving the one-hot vector with the widening_window.
    Inputs:
        - tempo: (torch.tensor) Tempo values for a batch. shape (batch_size, 1)
        - tempo_range: (tuple) Lower and upper bounds of linear tempo range: (tempo_min, tempo_max)
        - tempo_widening_win: (torch.Tensor) Widens one-hot vector to ±N adjacent tempo bins. Shape (2N+1). Bock's window is torch.Tensor([0.25,0.5,1.0,0.5,0.25]).'''

    all_tempi = np.arange(tempo_range[0], tempo_range[1])
    one_hot = torch.zeros(tempo.shape[0],len(all_tempi))
    for i, t in enumerate(tempo):
        if round(t.item()) < tempo_range[0] or round(t.item()) >= tempo_range[1]:
            tempo_index = 0
        else:
            try:
                tempo_index = np.where(all_tempi==int(round(t.item())))[0][0]
            except:
                raise ValueError("Error trying to find tempo_index. With t = {}, t.item() = {}, all_tempi = {} and np.where = {}".format(t,t.item(),all_tempi,np.where(all_tempi==int(round(t.item())))))
        # tempo_indexes.append(tempo_index)
        one_hot[i, tempo_index] = 1

    # Optionally apply widening window
    if not tempo_widening_win == None:
        one_hot = torch.unsqueeze(one_hot,1) # Add channel dimension
        padding = (tempo_widening_win.shape[0] - 1)//2
        # widened_onehot = np.convolve(one_hot, tempo_widening_win, mode="same")
        conv_tensor = torch.unsqueeze(torch.unsqueeze(tempo_widening_win,0),0)
        widened_onehot = torch.nn.functional.conv1d(one_hot,conv_tensor, padding=padding)
        widened_onehot = torch.squeeze(widened_onehot)
        return F.normalize(widened_onehot,p=1,dim=1)  # Normalise to a probability distribution. (makes it compatible with softmax)
    else:
        return one_hot

def tempo_to_class_index(tempo, tempo_range):
    '''Converts float tempo values into a class index - for pytorch cross-entropy loss.
    Assumes a linear tempo range. Note that the class index would be the index of the 1 on a one-hot vector.
    Inputs:
        - tempo: (torch.tensor) Tempo values for a batch. shape (batch_size, 1)
        - tempo_range: (tuple) Lower and upper bounds of linear tempo range: (tempo_min, tempo_max)
    Output:
        - class_indexes: (torch.Tensor) class indexes of the tempo value (i.e. index of the corresponding one-hot. shape (batch_size)'''
    all_tempi = np.arange(tempo_range[0], tempo_range[1])
    class_indexes = torch.empty(tempo.shape[0], dtype=torch.long)
    for i, t in enumerate(tempo):
        if round(t.item()) < tempo_range[0] or round(t.item()) >= tempo_range[1]:
            tempo_index = 0
        else:
            tempo_index = np.where(all_tempi == int(round(t.item())))[0][0]
        class_indexes[i] = tempo_index
    return class_indexes

def onehot_to_tempo(one_hots,tempo_range):
    '''Converts a batch of one-hot vectors into batch of float tempo values. Assumes a linear tempo range
    Inputs:
        - one_hots: (torch.tensor) One-hot vectors for a batch. shape (batch_size, one_hot_size)
        - tempo_range: (tuple) Lower and upper bounds of linear tempo range: (tempo_min, tempo_max)
    Output:
        - tempi: (torch.Tensor) Tempo values for a batch. Shape (batch_size, 1)'''

    all_tempi = torch.arange(tempo_range[0], tempo_range[1], dtype=torch.float)
    tempo_indexes = torch.argmax(one_hots,dim=1)
    tempi = all_tempi[tempo_indexes]
    tempi = torch.unsqueeze(tempi, 1)
    return tempi

def class_index_to_tempo(class_indexes, tempo_range):
    '''Converts a batch of class indexes into batch of float tempo values. Assumes a linear tempo range
        Inputs:
            - class_indexes: (torch.tensor, dtype=torch.long) Class indexes for a batch. shape (batch_size)
            - tempo_range: (tuple) Lower and upper bounds of linear tempo range: (tempo_min, tempo_max)
        Output:
            - tempi: (torch.Tensor) Tempo values for a batch. Shape (batch_size,1 )'''
    all_tempi = torch.arange(tempo_range[0], tempo_range[1])
    tempi = all_tempi[class_indexes]
    tempi = torch.unsqueeze(tempi, 1)
    return tempi

def class_index_to_onehot(class_indexes, tempo_range):
    '''Converts a batch of class indexes into batch of one-hot vectors. Assumes a linear tempo range
        Inputs:
            - class_indexes: (torch.tensor) Class indexes for a batch. shape (batch_size)
            - tempo_range: (tuple) Lower and upper bounds of linear tempo range: (tempo_min, tempo_max)
        Output:
            - one_hot: (torch.Tensor) One-hot vectors for a batch. Shape (batch_size, onehot_size)'''
    all_tempi = np.arange(tempo_range[0], tempo_range[1])
    one_hot = torch.zeros(class_indexes.shape[0], len(all_tempi))
    for n, idx in enumerate(class_indexes):
        one_hot[n, idx.item()] = 1
    return one_hot

def softmax_to_mirex(predictions, tempo_range=(0,300), sort=True):
    '''Get numerical tempi estimates from neural network predictions (softmax output). Outputs 2 tempi and relative weight per prediction, following MIREX format.
    This function assumes that the network output units represent linearly spaced tempi, with the range tempo_range.
    Inputs:
        - predictions: output of the network (vector)
        - tempo_range: linear spacing tempo range.
        - sort: if True sort the two tempi in ascending order. (by default we expect them to be sorted by descending strength)'''
    tempis = []
    for i in range(predictions.shape[0]):
        # Iterate over samples in the batch
        tempi_idx = torch.topk(predictions[i,:], 2,largest=True,sorted=True)[1]
        tempi = [np.array(range(tempo_range[0],tempo_range[1]))[tempi_idx[n].item()] for n in range(2)]
        weights = [ predictions[i,t].item() for t in tempi_idx]
        weight = weights[0]/sum(weights)
        # Sort tempi if necessary
        if sort==True:
            if tempi[0]<tempi[1]:
                tempis.append((tempi[0], tempi[1], weight))
            else:
                tempis.append((tempi[1], tempi[0], 1-weight))
        else:
            tempis.append((tempi[0],tempi[1],weight))
    return tempis

def tempo_to_mirex(tempi):
    '''Format predicted tempi in the MIREX format. Outputs 2 tempi and relative weight per prediction, following MIREX format.
        Inputs:
            - tempi: (torch.Tensor of floats) estimated float tempo values for a batch of tracks.
        Output:
            - tempi_mirex: (List of tuples) Mirex-formatted tempo estimates. Each tuple is (tempo1, tempo2, relative_weight).
            Because we don't have a second tempo estimate, set it to 0 and set relative weight to 1.0, i.e. for each track the tuple is (tempo, 0, 1.0)'''
    tempi_mirex = []
    for t in tempi:
        if t.item()>0:
            tempi_mirex.append( (t.item(), 0.0, 1.0) )
        else:
            tempi_mirex.append((0.5, 0.0, 1.0)) #MIR_eval needs a tempo > 0. If we don't meet that condition force tempo to 0.5.
    return tempi_mirex


def normatise_tempo(tempi, tempo_range=(0,300)):
    '''Normalise tempo float value to range [0,1].
    Force out-of-range values to 0.
    Inputs:
            - tempi: (torch.Tensor of floats) float tempo values for a batch of tracks.
        Output:
            - normalised_tempi: (torch.Tensor of floats) float normalised tempo values for a batch of tracks.'''
    normatised_tempi = (tempi - tempo_range[0])/(tempo_range[1] - tempo_range[0])
    # Force out-of-range values to 0
    normatised_tempi[normatised_tempi < tempo_range[0]] = 0.0
    normatised_tempi[normatised_tempi >= tempo_range[1]] = 0.0
    assert torch.max(normatised_tempi).item() <= 1.0, "Normalised tempo value go out of range [0,1], with max {}".format(torch.max(normatised_tempi).item())
    assert torch.min(normatised_tempi).item() >= 0.0, "Normalised tempo value go out of range [0,1], with min {}".format(torch.min(normatised_tempi).item())
    return normatised_tempi

def denormatise_tempo(normalised_tempi, tempo_range=(0,300)):
    '''De-normalise tempo float value from range [0,1] to tempo_range.
        Input:
            - normalised_tempi: (torch.Tensor of floats) float normalised tempo values for a batch of tracks.
        Output:
            - tempi: (torch.Tensor of floats) float tempo values for a batch of tracks.
        '''
    return normalised_tempi*(tempo_range[1] - tempo_range[0]) + tempo_range[0]