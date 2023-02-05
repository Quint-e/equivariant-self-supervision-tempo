# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Test augmentation layers

"""
import math

import torch
import torchaudio as ta

from sst.augmentations import TimeStretchFixedSize, Vol, PolarityInversion, GaussianNoise
# from sst.dataloader_audiofiles import DatasetAudioFiles
from sst.tests.utils import load_audio_batch

TEST_AUDIO_BATCH = ['../audio/1030011.LOFI.mp3','../audio/10258351.clip.mp3']


def make_mock_audio(sr=44100, len_s=6, n_channels=2, batch_size=8):
    audio = torch.clamp(torch.randn((batch_size, n_channels, sr * len_s)) / 2, min=-1.0, max=1.0)
    return audio

class TestTimeStretching:
    """Test TimeStretchingFixedSize augmentation class."""
    def test_y_invariance(self):
        # Set parameters to which the ground truth (y) should be invariant.
        rate = 1.0
        # Load audio
        x, sr = load_audio_batch(TEST_AUDIO_BATCH)
        print('audio mono',x.shape)
        # Make mock tempo ground truth
        tempo_range = (0, 300)
        y = (tempo_range[0] - tempo_range[1]) * torch.rand(len(TEST_AUDIO_BATCH)) + tempo_range[1]
        # Pre-process cx_spectrogram
        spectrogram_cx = ta.transforms.Spectrogram(
            n_fft=2048, 
            hop_length=441, 
            window_fn=torch.hann_window, 
            power=None
            )
        x = spectrogram_cx(x)
        # Apply augmentations
        aug = TimeStretchFixedSize(n_freq=1025,hop_length=441, tempo_range=tempo_range)
        x_aug, y_aug = aug(x, y, rate)
        print('x',x.shape)
        print('x_aug', x_aug.shape)
        # Test invariance
        assert torch.equal(y_aug,y), "y is affected by an invariant transformation."

    def test_y_equivariance(self):
        # Set parameters to which the ground truth (y) should be equivariant.
        rates = [0.8, 1.05, 1.2]
        # Load audio
        x, sr = load_audio_batch(TEST_AUDIO_BATCH)
        print('audio mono', x.shape)
        # Make mock tempo ground truth
        tempo_range = (0, 300)
        y = torch.Tensor([55,184.3])
        # Pre-process cx_spectrogram
        spectrogram_cx = ta.transforms.Spectrogram(
            n_fft=2048, 
            hop_length=441, 
            window_fn=torch.hann_window, 
            power=None
            )
        x = spectrogram_cx(x)
        for rate in rates:
            # Apply augmentations
            aug = TimeStretchFixedSize(n_freq=1025, hop_length=441,tempo_range=tempo_range)
            x_aug, y_aug = aug(x, y, rate)
            # Test that y has been modified
            assert not torch.equal(y_aug, y), "y is not affected by an equivariant transformation. with rate {}".format(rate)
            # Test equivariance.
            assert torch.equal(y*rate,y_aug), "y transformation is not equivariant. with rate {}".format(rate)

    def test_y_out_of_range(self):
        '''Verify that out-of range tempi that may emerge from time stretching are forced to tempo 0 (which encodes "no tempo")'''
        # Load audio
        x, sr = load_audio_batch(TEST_AUDIO_BATCH)
        print('audio mono', x.shape)
        # Pre-process cx_spectrogram
        spectrogram_cx = ta.transforms.Spectrogram(
            n_fft=2048, 
            hop_length=441, 
            window_fn=torch.hann_window, 
            power=None
            )
        x = spectrogram_cx(x)
        # Make mock tempo ground truth
        tempo_range = (30, 300)
        tempi = [0,1.2,350]
        rates = [1,0.8, 1.2]
        for tempo, rate in zip(tempi, rates):
            y = tempo * torch.ones(len(TEST_AUDIO_BATCH))
            # Apply augmentations
            aug = TimeStretchFixedSize(n_freq=1025, hop_length=441, tempo_range=tempo_range)
            x_aug, y_aug = aug(x, y, rate)
            assert torch.equal(y_aug, torch.zeros(len(TEST_AUDIO_BATCH))), "Out-of-range tempo should be set to 0 but got {}".format(y_aug)

    def test_x_invariance(self):
        '''Make sure the augmentations do not affect the size of the audio tensor'''
        # Set parameters to which the audio representation should be invariant.
        rate = 1.0
        # Load audio
        x, sr = load_audio_batch(TEST_AUDIO_BATCH)
        print('audio mono', x.shape)
        # Make mock tempo ground truth
        tempo_range = (0, 300)
        y = (tempo_range[0] - tempo_range[1]) * torch.rand(len(TEST_AUDIO_BATCH)) + tempo_range[1]
        # Pre-process cx_spectrogram
        spectrogram_cx = ta.transforms.Spectrogram(
            n_fft=2048, 
            hop_length=441, 
            window_fn=torch.hann_window, 
            power=None
            )
        x = spectrogram_cx(x)
        # Apply augmentations
        aug = TimeStretchFixedSize(n_freq=1025, hop_length=441, tempo_range=tempo_range)
        x_aug, y_aug = aug(x, y, rate)
        # Test Shape invariance
        assert x_aug.shape == x.shape, "The shape of x is affected by the TimeStretchFixedSize transformation."
        # Test invariance
        assert torch.equal(x_aug, x), "x is affected by an invariant transformation."

    def test_x_equivariant_transform(self):
        '''Test that the shape of x is not affected by an equivariant transformation but that its values are indeed modified'''
        # Set parameters to which the ground truth (y) should be equivariant.
        rates = [0.8, 1.05, 1.2]
        # Load audio
        x, sr = load_audio_batch(TEST_AUDIO_BATCH)
        print('audio mono', x.shape)
        # Make mock tempo ground truth
        tempo_range = (0, 300)
        y = (tempo_range[0] - tempo_range[1]) * torch.rand(len(TEST_AUDIO_BATCH)) + tempo_range[1]
        # Pre-process cx_spectrogram
        spectrogram_cx = ta.transforms.Spectrogram(n_fft=2048, hop_length=441, window_fn=torch.hann_window, power=None)
        x = spectrogram_cx(x)
        for rate in rates:
            # Apply augmentations
            aug = TimeStretchFixedSize(n_freq=1025, hop_length=441, tempo_range=tempo_range)
            x_aug, y_aug = aug(x, y, rate)
            # Test Shape invariance
            assert x_aug.shape == x.shape, "The shape of x is affected by the TimeStretchFixedSize transformation. with rate {}".format(rate)
            # Test that x has indeed been modified.
            assert not torch.equal(x_aug,x), "TimeStretchFixedSize transformation is not modifying x. with rate {}".format(rate)


class TestVol:
    '''Test custom volume augmentation class'''
    def test_range(self):
        '''Make sure the output is indeed in range[-1,1]'''
        audio = make_mock_audio()
        assert torch.max(audio).item() <= 1.0, "Max audio amplitude is > 1"
        assert torch.min(audio).item() >= -1.0, "Min audio amplitude is < -1"

    def test_invariance(self):
        '''Test that nothing changes when invariant gain is applied'''
        audio = make_mock_audio()
        # Test amplitude
        gain = 1.0
        vol = Vol(gain_type='amplitude')
        audio_aug_db = vol(audio, gain)
        assert torch.equal(audio, audio_aug_db), "Audio is affected by applying 1.0 (multiplicative) amplitude gain"
        # Test dB
        gain = 0.0
        vol = Vol(gain_type='db')
        audio_aug_db = vol(audio, gain)
        assert torch.equal(audio, audio_aug_db), "Audio is affected by applying 0dB gain"
        # Test Power
        gain = 1.0
        vol = Vol(gain_type='power')
        audio_aug_db = vol(audio, gain)
        assert torch.equal(audio, audio_aug_db), "Audio is affected by applying 0dB (power) gain"

    def test_amplitude(self):
        '''Test that we get the expected audio output.'''
        gains = [1.22, 0.005, 0.2, 3.8, 0.0]
        audio = make_mock_audio()
        vol = Vol(gain_type='amplitude')
        for gain in gains:
            audio_aug = vol(audio, gain)
            audio_aug_expected = torch.clamp(audio*gain, -1, 1)
            assert torch.equal(audio_aug, audio_aug_expected)

    def test_dB(self):
        '''Test that we get the expected audio output.'''
        gains = [1.22, -3, 0.2, 0.0]
        audio = make_mock_audio()
        vol = Vol(gain_type='db')
        for gain in gains:
            audio_aug = vol(audio, gain)
            audio_aug_expected = torch.clamp(ta.functional.gain(audio, gain), -1, 1)
            assert torch.equal(audio_aug, audio_aug_expected)

    def test_power(self):
        '''Test that we get the expected audio output.'''
        gains = [0.2, 3, 2, 4.5]
        audio = make_mock_audio()
        vol = Vol(gain_type='power')
        for gain in gains:
            audio_aug = vol(audio, gain)
            audio_aug_expected = torch.clamp(ta.functional.gain(audio, 10 * math.log10(gain)), -1, 1)
            assert torch.equal(audio_aug, audio_aug_expected)


class TestPolarityInversion:
    def test_pol_inv(self):
        audio = make_mock_audio()
        pol_inv = PolarityInversion()
        audio_inv = pol_inv(audio)
        assert audio.shape == audio_inv.shape, "Polarity inversion modifies audio shape"
        assert torch.equal(audio, -1.0*audio_inv)


class TestGaussianNoise:
    def test_shape(self):
        audio = make_mock_audio()
        gauss_noise = GaussianNoise()
        std = torch.std(audio)
        noise_std = 0.5 * std
        audio_noisy = gauss_noise(audio, noise_std)
        assert audio.shape==audio_noisy.shape

    def test_invariance(self):
        audio = make_mock_audio()
        gauss_noise = GaussianNoise()
        noise_std = 0.0
        audio_noisy = gauss_noise(audio, noise_std)
        assert torch.equal(audio, audio_noisy), "Adding zero std noise modifies audio"


if __name__ == "__main__":
    test_timestretching = TestTimeStretching()
    test_timestretching.test_y_invariance()
    test_timestretching.test_y_equivariance()
    test_timestretching.test_y_out_of_range()
    test_timestretching.test_x_invariance()
    test_timestretching.test_x_equivariant_transform()

    test_vol = TestVol()
    test_vol.test_range()
    test_vol.test_invariance()
    test_vol.test_amplitude()
    test_vol.test_dB()
    test_vol.test_power()

    test_pol_inv = TestPolarityInversion()
    test_pol_inv.test_pol_inv()

    test_gauss_noise = TestGaussianNoise()
    test_gauss_noise.test_shape()
    test_gauss_noise.test_invariance()