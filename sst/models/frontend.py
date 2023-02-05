# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Pre-processing front-end for all neural networks.
"""
import torch
import torchaudio as ta
import torch.nn as nn

from sst.augmentations import TimeStretchFixedSize, Vol, PolarityInversion, GaussianNoise

EPS = 10e-8


class AugParams:
    pass


class FrontEndAug(nn.Module):
    '''Main module class'''
    def __init__(self, config, tempo_range=(0, 300)):
        super(FrontEndAug, self).__init__()
        self.config = config #This is only the "frontend" config - i.e. a subset of the full reference config.
        self.tempo_range = tempo_range
        self.freq_mask_param = self.config.n_fft//(1/self.config.aug_params.freq_masking.mask_ratio_max)

        # Time domain transforms
        self.vol = Vol(gain_type='db')
        self.pol_inv = PolarityInversion()
        self.gauss_noise = GaussianNoise()

        # Complex Spectrogram Pre-processing
        self.spectrogram_cx = ta.transforms.Spectrogram(
            n_fft=self.config.n_fft, 
            hop_length=self.config.hop_length, 
            window_fn=torch.hann_window,
            power=None
            )
        self.melscale = ta.transforms.MelScale(
            sample_rate=self.config.sr, 
            n_stft=self.config.n_fft//2+1, 
            f_min=self.config.f_min, 
            f_max=self.config.f_max, 
            n_mels=self.config.n_mels
            )

        # Frequency domain transforms
        self.timestretch = TimeStretchFixedSize(
            hop_length=self.config.hop_length, 
            n_freq=self.config.n_fft//2+1, 
            tempo_range=self.tempo_range
            )
        self.freq_masking = ta.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param)

    def draw_timestretch_rate(self, rate_min=0.8, rate_max=1.2):
        rate_tensor = (rate_min - rate_max)*torch.rand(1) + rate_max
        return rate_tensor.item()

    def draw_random_float_in_range(self, value_min: float, value_max: float):
        '''Draw a random float value in range [value_min, value_max]'''
        value_tensor = (value_min - value_max)*torch.rand(1) + value_max
        return value_tensor.item()

    def forward(self, x, y):
        # Waveform Augmentations
        if 'volume' in self.config.augmentations:
            vol_gain = self.draw_random_float_in_range(
                self.config.aug_params.volume.gain_min, 
                self.config.aug_params.volume.gain_max
                )
            x = self.vol(x, vol_gain)
        if 'polarity_inversion' in self.config.augmentations:
            p = self.draw_random_float_in_range(0, 1)
            if p <= self.config.aug_params.polarity_inversion.prob:
                x = self.pol_inv(x)
        if 'gaussian_noise' in self.config.augmentations:
            std = self.draw_random_float_in_range(
                self.config.aug_params.gaussian_noise.std_min, 
                self.config.aug_params.gaussian_noise.std_min
                )
            x = self.gauss_noise(x, std)
        # Complex spectrogram
        x = self.spectrogram_cx(x)
        if 'timestretch' in self.config.augmentations:
            ts_rate = self.draw_random_float_in_range(
                self.config.aug_params.timestretch.rate_min, 
                self.config.aug_params.timestretch.rate_max
                )
            x, y = self.timestretch(x, y, overriding_rate=ts_rate)
        else:
            ts_rate = -1
        if self.config.power is not None:
            if self.config.power == 1.0:
                x = x.abs()
            else:
                x = x.abs().pow(self.config.power)
        if 'freq_masking' in self.config.augmentations:
            x = self.freq_masking(x)
        # Compute log melspectrogram
        x = self.melscale(x)
        x = torch.log(x + EPS)
        return x, y, ts_rate


class FrontEndNoAug(nn.Module):
    '''Main module class'''
    def __init__(self, config):
        super(FrontEndNoAug, self).__init__()
        self.config = config
        # audio Pre-processing
        self.spectrogram_cx = ta.transforms.Spectrogram(
            n_fft=self.config.n_fft, 
            hop_length=self.config.hop_length,
            window_fn=torch.hann_window, 
            power=None, 
            return_complex=True
            )
        self.melscale = ta.transforms.MelScale(
            sample_rate=self.config.sr, 
            n_stft=self.config.n_fft // 2 + 1,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            n_mels=self.config.n_mels
            )

    def forward(self, x, y):
        x = self.spectrogram_cx(x)
        # Get magnitude / power spectrogram
        if self.config.power is not None:
            if self.config.power == 1.0:
                x = x.abs()
            else:
                x = x.abs().pow(self.config.power)
        # Compute log melspectrogram
        x = self.melscale(x)
        x = torch.log(x + EPS)
        # Mock ts_rate, for consistency with FrontEndAug class.
        ts_rate = -1
        return x, y, ts_rate
