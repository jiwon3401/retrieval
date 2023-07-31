import time
import os
from itertools import chain
import random

import torch
import torch.nn as nn
import torchaudio

import h5py
import numpy as np
import librosa
from re import sub
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from tools.file_io import load_csv_file, write_pickle_file
from scipy.io import wavfile
import matplotlib.pyplot as plt

import math
import pathlib
from tqdm import tqdm
import soundfile as sf

from tools.dataset import load_metadata, pad_or_truncate, _create_vocabulary, _sentence_process



#raw audio augmentation
#RandomClip, Randomspeedchange, RandomBackgroundNoise

class RandomClip:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = clip_length
        #self.sequence_length = sequence_length

        self.vad = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[0]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length-self.clip_length)
            audio_data = audio_data[offset:(offset+self.clip_length)]

        return self.vad(audio_data) # remove silences at the beggining/end
    


class RandomSpeedChange:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1])
        if speed_factor == 1.0: # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio



class RandomBackgroundNoise:
    '''
    pick a random noise file from a given folder and will apply it to the original audio file.
    '''

    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))

        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

            
    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ['remix', '1'], # convert to mono
            ['rate', str(self.sample_rate)], # resample
        ]

        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]

        if noise_length > audio_length:
            offset = random.randint(0, noise_length-audio_length)
            noise = noise[..., offset:offset+audio_length]

        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length-noise_length))], dim=-1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise ) / 2


    

class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data

    

compose_transform = ComposeTransform([
    RandomClip(sample_rate=sampling_rate, clip_length=64000),
    RandomSpeedChange(sampling_rate),
    RandomBackgroundNoise(sampling_rate, './data/musan/noise')])



#path 지정
data_path = '/home/clim-lab/바탕화면/main_drive/jiwon/retrieval/data/Clotho'
csv_path = os.path.join(data_path, 'csv_files/')
train_csv_path = os.path.join(csv_path, 'train.csv')
train_audio_dir = os.path.join(data_path, 'waveforms/train/')
train_hdf5_path = os.path.join(data_path, 'hdf5s/train/train.h5')
dataset='Clotho'
train_meta_dict = load_metadata(dataset, train_csv_path)



def augment_raw_audio(dataset, train_meta_dict):
    '''
    making raw audio augmentation using randomclip, randomnoise, randomspeedchange
    file saved in './data/Clotho/waveforms/
    '''

    output_augment_path = os.path.join(data_path, 'waveforms/train_augment')
    audio_nums = len(train_meta_dict['audio_name'])

    for i in tqdm(range(audio_nums)):
        audio_name = train_meta_dict['audio_name'][i]

        #using torchaudio.load
        audio, sr = librosa.load(train_audio_dir + audio_name, sr=sampling_rate, mono=True)
        audio, audio_length = pad_or_truncate(audio, max_audio_length)
                                 
        audio_trans = torch.Tensor(audio.reshape(1,-1))
        
        new_filename = audio_name[:-4] + '_aug.wav'
        output_path = os.path.join(output_augment_path, new_filename)


        #audio augmentation 3가지 방법 모두 적용
        transformed_audio = compose_transform(audio_trans)
        transformed_audio_numpy = transformed_audio.numpy()[0] 
        #torchaudio.load() returns a 2-dimensional tensor, select the first channel.

        sf.write(output_path, transformed_audio_numpy, 32000, format='WAV')
        
        

        
##################################################################################
#Specaugment-> applied directly to the feature inputs of a neural network.
#from torchlibrosa.augmentation import SpecAugmentation
def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    
    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec


#spec_augment(x)