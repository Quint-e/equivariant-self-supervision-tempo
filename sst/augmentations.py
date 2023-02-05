# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Augmentation layers

"""
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta

class TimeStretchFixedSize(nn.Module):
    """
    THIS CLASS IS NOT ACTUALLY USED IN TRAINING. PLEASE IGNORE
    Applies time stretching and preserves the input dimension (via padding or slicing).
    - tempo_range: (min_tempo, max_tempo) linear tempo range"""
    def __init__(
            self, 
            n_freq=1025, 
            hop_length=441, 
            tempo_range=(0,300)
            ):
        super(TimeStretchFixedSize, self).__init__()
        self.n_freq = n_freq
        self.hop_length = hop_length

        self.tempo_range = tempo_range

        # Time-stretching
        self.timestretch = ta.transforms.TimeStretch(
            hop_length=self.hop_length, 
            n_freq=self.n_freq, 
            fixed_rate=None
            )

    def resize_after_timestretch(self,x, original_shape):
        '''Resize tensor so that it has the same shape before and after being time-stretched. If longer (because slowed down) only keep a chunk. If shorter, pad with zeros.'''
        time_stretched_len = x.shape[3]
        original_len = original_shape[3]
        if time_stretched_len == original_len:
            return x
        else:
            if time_stretched_len > original_len:
                return x[:,:,:,:original_len]
            else:
                pad_len = original_len - time_stretched_len
                return F.pad(x,(0,pad_len), "constant", 0)

    def timestretch_and_resize(self,x, overriding_rate):
        '''Time-stretch and resize the audio complex spectrogram'''
        original_shape = x.shape
        x = self.timestretch(complex_specgrams=x, overriding_rate=overriding_rate)
        x = self.resize_after_timestretch(x,original_shape)
        assert x.shape==original_shape, "time-stretching is modifying tensor size"
        return x

    def transform_y(self,y,overriding_rate):
        '''Modify the target vector so that the ground truth reflects the transformation applied to the audio.
        Assumes tempo float value.
        Inputs:
            - y: ground truth tempo. Float tensor of dimention (batch_size).
            - overriding_rate: (float) speed=up rate to apply'''
        y = y*overriding_rate
        # Make sure the tempo after time stretching does not go out of range.
        y[y < self.tempo_range[0]] = 0.0
        y[y > self.tempo_range[1]] = 0.0
        return y

    def forward(self, x, y, overriding_rate=1.0):
        '''Expects a complex spectrogram as input'''
        x = self.timestretch_and_resize(x, overriding_rate=overriding_rate)
        y = self.transform_y(y,overriding_rate)
        return x, y


class Vol(torch.nn.Module):
    r"""Add a volume to an waveform.

    Args:

        gain_type (str, optional): Type of gain. One of: ``amplitude``, ``power``, ``db`` (Default: ``amplitude``)
    """

    def __init__(self,  gain_type: str = 'amplitude'):
        super(Vol, self).__init__()
        self.gain_type = gain_type
        if not self.gain_type in ['amplitude', 'power', 'db']:
            raise ValueError("gain_type must be one of ['amplitude', 'power', 'db'], but got {}".format(self.gain_type))


    def forward(self, waveform: Tensor, gain: float) -> Tensor:
        r"""
        Modified torchaudio.transforms.vol class, so that the gain can be passed in the forward method, as opposed to be fixed at the instance level.
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).
            gain (float): Interpreted according to the given gain_type:
            If ``gain_type`` = ``amplitude``, ``gain`` is a positive amplitude ratio.
            If ``gain_type`` = ``power``, ``gain`` is a power (voltage squared).
            If ``gain_type`` = ``db``, ``gain`` is in decibels.

        Returns:
            Tensor: Tensor of audio of dimension (..., time).
        """
        if self.gain_type in ['amplitude', 'power'] and gain < 0:
            raise ValueError("If gain_type = amplitude or power, gain must be positive but got {}.".format(gain))

        if self.gain_type == "amplitude":
            waveform = waveform * gain

        if self.gain_type == "db":
            waveform = ta.functional.gain(waveform, gain)

        if self.gain_type == "power":
            waveform = ta.functional.gain(waveform, 10 * math.log10(gain))

        return torch.clamp(waveform, -1, 1)


class PolarityInversion(nn.Module):
    """Inverts the polarity of the raw audio signal."""
    def __init__(self):
        super(PolarityInversion, self).__init__()

    def forward(self, waveform: Tensor) -> Tensor:
        r"""
        Invert the polarity of audio by multiplying by -1
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).
        Returns:
            Tensor: Tensor of audio of dimension (..., time).
        """
        return waveform * -1.0

class GaussianNoise(nn.Module):
    def __init__(self):
        """
        Add Gaussian noise of mean 0.0 and adjustable standard deviation (i.e. loudness) to audio signal.
        """
        super(GaussianNoise, self).__init__()

    def forward(self, audio: Tensor, noise_std: float) -> Tensor:
        if noise_std <0:
            raise ValueError("Gaussian noise std should be a positive float")
        noise = torch.randn(audio.shape, device=audio.device) * noise_std
        return audio + noise



