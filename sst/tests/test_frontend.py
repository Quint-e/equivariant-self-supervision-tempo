# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Test augmentation layers

"""
import torch
import torchaudio as ta
import torch.nn as nn
from omegaconf import OmegaConf

from sst.models.frontend import FrontEndAug, FrontEndNoAug, EPS
from sst.tests.utils import load_audio_batch

TEST_AUDIO_BATCH = ['../audio/1030011.LOFI.mp3','../audio/10258351.clip.mp3']


class TaMelspec(nn.Module):
    def __init__(self):
        super(TaMelspec, self).__init__()
        self.melspec = ta.transforms.MelSpectrogram(
            sample_rate=44100, 
            n_fft=2048, 
            hop_length=441,
            window_fn=torch.hann_window, 
            f_min=30, 
            f_max=17000, 
            n_mels=81, 
            power=1
            )

    def forward(self,x):
        x = self.melspec(x)
        x = torch.log(x + EPS)
        return x


class TestFrontEndAug:
    def test_melspec_equivalence(self):
        '''Test that the audio pre-processing transform produces same output as torchaudio.MelSpectrogram when no augmentation'''
        # Load audio
        x, sr = load_audio_batch(TEST_AUDIO_BATCH, num_samples=600000)
        print('audio mono', x.shape)
        # Make mock tempo ground truth
        tempo_range = (0,300)
        y = (tempo_range[0] - tempo_range[1]) * torch.rand(len(TEST_AUDIO_BATCH)) + tempo_range[1]
        # Process melspec with torchaudio
        ta_mel = TaMelspec()
        melspec_ta = ta_mel(x)
        # Process melspec with frontend
        frontend_config_yaml = '''
            sr: 44100
            n_fft: 2048
            hop_length: 441
            n_mels: 81
            f_min: 30
            f_max: 17000
            power: 1
            augmentations:
                - 
            aug_params:
                timestretch:
                    rate_min: 0.8
                    rate_max: 1.2
                volume:
                    gain_min: -5.0
                    gain_max: 5.0
                polarity_inversion:
                    prob: 0.5
                gaussian_noise:
                    std_min: 0.0
                    std_max: 0.7
                freq_masking:
                    mask_ratio_max: 0.2
        '''
        frontend_config = OmegaConf.create(frontend_config_yaml)
        frontend_aug = FrontEndAug(frontend_config)
        melspec_fe, y, ts_rate = frontend_aug(x, y)
        print('melspec_ta', melspec_ta.shape)
        print('melspec_fe', melspec_fe.shape)
        assert torch.equal(melspec_ta,melspec_fe), "No augmentation fronted is not equivalent to torchaudio.MelSpectrogram"

    def test_melspec_shape_equivalence(self):
        '''Test that the audio pre-processing transform produces output of same shape as torchaudio.MelSpectrogram when no augmentation'''
        # Load audio
        x, sr = load_audio_batch(TEST_AUDIO_BATCH, num_samples=600000)
        print('audio mono', x.shape)
        # Make mock tempo ground truth
        tempo_range = (0,300)
        y = (tempo_range[0] - tempo_range[1]) * torch.rand(len(TEST_AUDIO_BATCH)) + tempo_range[1]
        # Process melspec with torchaudio
        ta_mel = TaMelspec()
        melspec_ta = ta_mel(x)
        # Process melspec with frontend_no_aug
        frontend_config_yaml = '''
            sr: 44100
            n_fft: 2048
            hop_length: 441
            n_mels: 81
            f_min: 30
            f_max: 17000
            power: 1
            augmentations:
                - 
            aug_params:
                timestretch:
                    rate_min: 0.8
                    rate_max: 1.2
                volume:
                    gain_min: -5.0
                    gain_max: 5.0
                polarity_inversion:
                    prob: 0.5
                gaussian_noise:
                    std_min: 0.0
                    std_max: 0.7
                freq_masking:
                    mask_ratio_max: 0.2
        '''
        frontend_config = OmegaConf.create(frontend_config_yaml)
        frontend_aug = FrontEndAug(frontend_config)
        melspec_fe, y, ts_rate = frontend_aug(x, y)
        print('melspec_ta', melspec_ta.shape)
        print('melspec_fe', melspec_fe.shape)
        assert melspec_ta.shape==melspec_fe.shape, "Augmentation fronted does not equivalent tensor shape to torchaudio.MelSpectrogram"


class TestFrontEndNoAug:
    def test_melspec_equivalence(self):
        '''Test that the audio pre-processing transform produces same output as torchaudio.MelSpectrogram'''
        # Load audio
        x, sr = load_audio_batch(TEST_AUDIO_BATCH, num_samples=600000)
        print('audio mono', x.shape)
        # Make mock tempo ground truth
        tempo_range = (0,300)
        y = (tempo_range[0] - tempo_range[1]) * torch.rand(len(TEST_AUDIO_BATCH)) + tempo_range[1]
        # Process melspec with torchaudio
        ta_mel = TaMelspec()
        melspec_ta = ta_mel(x)
        # Process melspec with frontend_no_aug
        frontend_config_yaml = '''
            sr: 44100
            n_fft: 2048
            hop_length: 441
            n_mels: 81
            f_min: 30
            f_max: 17000
            power: 1
        '''
        frontend_config = OmegaConf.create(frontend_config_yaml)
        frontend_no_aug = FrontEndNoAug(frontend_config)
        melspec_fe, y, ts_rate = frontend_no_aug(x, y)
        print('melspec_ta', melspec_ta.shape)
        print('melspec_fe', melspec_fe.shape)
        assert torch.equal(melspec_ta,melspec_fe), "No augmentation fronted is not equivalent to torchaudio.MelSpectrogram"



if __name__ == "__main__":
    test_frontend_aug = TestFrontEndAug()
    test_frontend_aug.test_melspec_equivalence()
    test_frontend_aug.test_melspec_shape_equivalence()
    test_fronted_no_aug = TestFrontEndNoAug()
    test_fronted_no_aug.test_melspec_equivalence()