# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Test utils functions

"""
import torch
import torchaudio as ta


def _load_audio(audio_filepath, num_frames=0):
    audio, sr = ta.load(audio_filepath, num_frames=num_frames)
    return audio, sr

def load_audio_batch(audio_filepaths, num_samples=200000):
    audio_tensors = []
    for audio_filepath in audio_filepaths:
        x, sr = _load_audio(audio_filepath, num_frames=num_samples)
        x = torch.mean(x, dim=0)
        x = torch.unsqueeze(x, 0)  # add Channel dimension
        audio_tensors.append(x)
    x = torch.stack(audio_tensors,dim=0)
    return x, sr
