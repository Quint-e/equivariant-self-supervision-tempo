# -*- coding: utf-8 -*-
"""
@author: Elio Quinton
python3.8

Torchaudio.info seems to be reporting an erroneous estimate of audio length.

"""
import os
import json
from time import time
import random
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio as ta


MP3_PAD = 8192 # Mp3 decoding does not specify the exact length of decoded audio, so we may get errors when trying to extract and excerpt close to the end of the file. This pad allows to stay away from the end of the file


class DatasetAudioFiles(Dataset):
    '''Loads audio directly from audio files stored on disk.
    If excerpt_len=None, the full track is returned'''
    def __init__(self, config):
        self.tempo_range = (0,300)
        self.config = config
        print('Dataloader config', self.config.__dict__)
        # Set a variable to agdregate all the tempo data and filepaths from multiple json indexes.
        self.dataset_index = []
        # For each dataset index file, get all filepaths and tempi.
        for path in self.config.indexes:
            with open(path, 'r') as f:
                di = json.load(f)
            self.dataset_index.extend(di)
        # Shuffle order of tracks (which mixes different datasets)
        random.seed(self.config.rseed)
        random.shuffle(self.dataset_index)

    def _load_audio(self,filepath):
        ######## Load audio of appropriate length ############
        time_1 = time()
        if self.config.num_samples==None:
            if self.config.random_excerpt==True:
                raise ValueError("You must specify num_samples if setting random_excerpt=True.")
            # Load full track
            audio, sr = ta.load(filepath, frame_offset=0)
        else:
            metadata = ta.info(filepath)  # get signal info to know audio length before loading
            audio_len = metadata.num_frames - MP3_PAD
            # Make sure we get the right number of samples depending on target sampling rate (the audio may get resampled)
            num_samples = int(self.config.num_samples * metadata.sample_rate // self.config.sr)
            if self.config.random_excerpt==True:
                if audio_len >= num_samples:
                    start_sample = random.randrange(0, audio_len - num_samples)
                    # print("audio_len", audio_len)
                    # print("start sample", start_sample)
                    audio, sr = ta.load(filepath, frame_offset=start_sample, num_frames=num_samples)
                else:
                    #Load full audio and actual audio zero-pad to expected length
                    print("SHORT TRACK")
                    audio, sr = ta.load(filepath, frame_offset=0)
                    if audio.shape[1]<num_samples:
                        pad_len = num_samples - audio.shape[1]
                        audio = F.pad(audio, (0,pad_len), "constant", 0)
                    else:
                        # In case actual audio is longer than num_samples in case of bad track length estimation from torchaudio.info
                        audio = audio[:,:num_samples]
            else:
                # Load audio segment from specified start_sample
                audio, sr = ta.load(filepath, frame_offset=self.config.start_sample, num_frames=num_samples)

        time_2 = time()
        ########## Downmix if necessary ################
        if self.config.downmix_to_mono:
            audio = torch.mean(audio, dim=0)
        ########## Resample if necessary ################
        resample = ta.transforms.Resample(orig_freq=sr, new_freq=self.config.sr)
        if sr != self.config.sr:
            audio = resample.forward(audio)
        time_3=time()
        return audio

    def _draw_random_float_in_range(self, value_min: float, value_max: float):
        '''Draw a random float value in range [value_min, value_max]'''
        value_tensor = (value_min - value_max)*torch.rand(1) + value_max
        return value_tensor.item()

    def _draw_sox_effects(self):
        '''Draw random parameters for active effects'''
        effects = []
        if 'pitch_shifting' in self.config.augmentations:
            cents = self._draw_random_float_in_range(self.config.aug_params.pitch_shifting.cent_min, self.config.aug_params.pitch_shifting.cent_max)
            effects.append(['pitch', str(cents)])
            effects.append(['rate', str(self.config.sr)])
        if 'timestretch' in self.config.augmentations:
            ts_rate = self._draw_random_float_in_range(self.config.aug_params.timestretch.rate_min, self.config.aug_params.timestretch.rate_max)
            effects.append(['tempo', '-m', str(ts_rate)])
        else:
            ts_rate = -1
        return effects, ts_rate

    def _shape_padding(self, tensor: torch.Tensor, target_shape: torch.Size):
        '''Make sure the tensor has the target_shape. Force it into it if not.
        This function is necessary because Sox resample is not sample-accurate: it sometimes adds or deletes a sample.'''
        pad_len = target_shape[1] - tensor.shape[1]
        if pad_len == 0:
            return tensor
        else:
            if pad_len < 0 :
                return tensor[:, :target_shape[1]]
            else :
                return F.pad(tensor, (0,pad_len), "constant", 0)

    def _apply_sox_effects(self, tensor: torch.Tensor, sample_rate: int, effects: List[List[str]]) -> torch.Tensor:
        shape_in = tensor.shape
        tensor, sr = ta.sox_effects.apply_effects_tensor(tensor, sample_rate, effects)
        tensor = self._shape_padding(tensor, shape_in)
        assert sample_rate==sr, "Sox effect modified sample rate. Was {} and now {}".format(sample_rate, sr)
        assert shape_in == tensor.shape, "Sox effect modified tensor shape. Was {} and now {}".format(shape_in, tensor.shape)
        return tensor

    def transform_y(self, y: torch.Tensor, ts_rate: float) -> torch.Tensor:
        '''Modify the target vector so that the ground truth reflects the transformation applied to the audio.
        Assumes tempo float value.
        Inputs:
            - y: ground truth tempo. Float tensor of dimention (batch_size).
            - overriding_rate: (float) speed=up rate to apply'''
        y = y*ts_rate
        # Make sure the tempo after time stretching does not go out of range.
        y[y < self.tempo_range[0]] = 0.0
        y[y > self.tempo_range[1]] = 0.0
        return y

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, idx):
        example_obj = self.dataset_index[idx]
        filepath = os.path.join(self.config.basedir,example_obj["audio_filepath"])
        tempo = torch.Tensor([example_obj["tempo"]])
        audio = self._load_audio(filepath)
        audio.unsqueeze_(0) #add a channel dimension
        if self.config.use_augmentations==True:
            sox_effects, ts_rate = self._draw_sox_effects()
            if sox_effects != []:
                audio = self._apply_sox_effects(audio, self.config.sr, sox_effects)
            if ts_rate != -1:
                tempo = self.transform_y(tempo, ts_rate)
        else:
            ts_rate = -1
        return (audio, tempo, ts_rate)


class DatasetDualAug(DatasetAudioFiles):
    '''Loads two copies of each audio file, with two different augmentations.'''

    def __getitem__(self, idx):
        # Check that that some activations are requested.
        if self.config.use_augmentations==False:
            return ValueError('This data loader is designed to be used with augmentations, but use_augmentation is not set to True in the dataset config')
        if len(self.config.augmentations) < 1:
            return ValueError('No augmentation specified. This data loader is designed to be used with augmentations, please specify at least one augmentation')
        # Load data
        example_obj = self.dataset_index[idx]
        filepath = os.path.join(self.config.basedir,example_obj["audio_filepath"])
        tempo = torch.Tensor([example_obj["tempo"]])
        audio = self._load_audio(filepath)
        audio.unsqueeze_(0) #add a channel dimension
        # Augmentations - set 1
        sox_effects_i, ts_rate_i = self._draw_sox_effects()
        sox_effects_j, ts_rate_j = self._draw_sox_effects()
        if sox_effects_i != []:
            audio_i = self._apply_sox_effects(audio, self.config.sr, sox_effects_i)
        if sox_effects_j != []:
            audio_j = self._apply_sox_effects(audio, self.config.sr, sox_effects_j)
        if ts_rate_i != -1:
            tempo_i = self.transform_y(tempo, ts_rate_i)
        if ts_rate_j != -1:
            tempo_j = self.transform_y(tempo, ts_rate_j)
        return (audio_i, tempo_i, ts_rate_i), (audio_j, tempo_j, ts_rate_j)

