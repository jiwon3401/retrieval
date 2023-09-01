import time
from itertools import chain
import os
import random
import math
import torch.nn as nn
import torchaudio
import h5py
import pathlib
import numpy as np
import librosa

from re import sub
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from scipy.io import wavfile
import matplotlib.pyplot as plt
import soundfile as sf

from tools.file_io import load_csv_file, write_pickle_file
from tools.dataset import load_metadata, _create_vocabulary, _sentence_process, pad_or_truncate
from tools.audio_augment import RandomClip, RandomSpeedChange, RandomBackgroundNoise
from tools.audio_augment import augment_raw_audio, pack_augment_dataset_to_hdf5
#from tools.text_augment import 


#path 지정
data_path = '/home/user/jiwon/retrieval/data/Clotho'
csv_path = os.path.join(data_path, 'csv_files/')
train_csv_path = os.path.join(csv_path, 'train.csv')
train_audio_dir = os.path.join(data_path, 'waveforms/train/')
train_hdf5_path = os.path.join(data_path, 'hdf5s/train/train.h5')
dataset='Clotho'
train_meta_dict = load_metadata(dataset, train_csv_path)
sampling_rate = 32000


if __name__ == '__main__':
    # logger.info('Creating Clotho augmentation files...')
    # augment_raw_audio(dataset, train_meta_dict)
    # logger.info('Creating raw audio DONE!')
    
    # logger.info('Packing Audio Augmented Clotho data...')
    # pack_augment_dataset_to_hdf5('Clotho')
    # logger.info('augmented Clotho done!')


    # logger.info('Text file augmentation')
    # logger.info('Packing Text Augmented Clotho data...')
    # pack_text_augment_dataset_to_hdf5('Clotho')
    # logger.info('augmented Clotho text data done!')