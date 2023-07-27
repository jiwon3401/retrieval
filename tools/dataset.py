import time
from itertools import chain

import h5py
import numpy as np
import librosa
from re import sub
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from tools.file_io import load_csv_file, write_pickle_file

from tools.audio_augment import spec_augment
#from audio_augmentations import *




def load_metadata(dataset, csv_file):
    """Load meta data of Clotho
    """
    if dataset == 'AudioCaps' and 'train' in csv_file:
        caption_field = None
    else:
        caption_field = ['caption_{}'.format(i) for i in range(1, 6)]
    csv_list = load_csv_file(csv_file)

    audio_names = []
    captions = []

    for i, item in enumerate(csv_list):

        audio_name = item['file_name']
        if caption_field is not None:
            item_captions = [_sentence_process(item[cap_ind], add_specials=False) for cap_ind in caption_field]
            # csv file 정제 -> lowercase, 
        else:
            item_captions = _sentence_process(item['caption'])
        audio_names.append(audio_name)
        captions.append(item_captions)

    meta_dict = {'audio_name': np.array(audio_names), 'captions': np.array(captions)}

    return meta_dict


def pack_dataset_to_hdf5(dataset):
    """

    Args:
        dataset: 'AudioCaps', 'Clotho'

    Returns:

    """

    splits = ['train', 'val', 'test']
    sampling_rate = 32000
    all_captions = [] 
    #csv file에 있는 caption들을 list로 한번에 저장 ex) train data -> 3839*5 = 19195개 caption 저장
    
    #if dataset == 'AudioCaps':
    #    audio_duration = 10
    if dataset == 'Clotho':
        audio_duration = 30
        
    else:
        raise NotImplementedError(f'No dataset named: {dataset}')

    max_audio_length = audio_duration * sampling_rate # 30 * 32000
    
    
    for split in splits:
        csv_path = 'data/{}/csv_files/{}.csv'.format(dataset, split)
        audio_dir = 'data/{}/waveforms/{}/'.format(dataset, split)
        hdf5_path = 'data/{}/hdf5s/{}/'.format(dataset, split)

        # make dir for hdf5
        Path(hdf5_path).mkdir(parents=True, exist_ok=True)

        meta_dict = load_metadata(dataset, csv_path)
        # meta_dict: {'audio_names': [], 'captions': []}

        audio_nums = len(meta_dict['audio_name'])

        if split == 'train':
            # store all captions in training set into a list
            if dataset == 'Clotho':
                for caps in meta_dict['captions']:
                    for cap in caps:
                        all_captions.append(cap)
            else:
                all_captions.extend(meta_dict['captions'])

        start_time = time.time()
        

        with h5py.File(hdf5_path+'{}.h5'.format(split), 'w') as hf:

            hf.create_dataset('audio_name', shape=(audio_nums,), dtype=h5py.special_dtype(vlen=str))
            hf.create_dataset('audio_length', shape=(audio_nums,), dtype=np.uint32)
            hf.create_dataset('waveform', shape=(audio_nums, max_audio_length), dtype=np.float32)

            if split == 'train' and dataset == 'AudioCaps':
                hf.create_dataset('caption', shape=(audio_nums,), dtype=h5py.special_dtype(vlen=str))
            
            else: #'clotho dataset'
                hf.create_dataset('caption', shape=(audio_nums, 5), dtype=h5py.special_dtype(vlen=str))

            for i in tqdm(range(audio_nums)):
                audio_name = meta_dict['audio_name'][i]

                audio, _ = librosa.load(audio_dir + audio_name, sr=sampling_rate, mono=True)
                audio, audio_length = pad_or_truncate(audio, max_audio_length)



                hf['audio_name'][i] = audio_name.encode()
                hf['audio_length'][i] = audio_length
                hf['waveform'][i] = audio
                hf['caption'][i] = meta_dict['captions'][i]

        logger.info(f'Packed {split} set to {hdf5_path} using {time.time() - start_time} s.')
    
    words_list, words_freq = _create_vocabulary(all_captions)
    # 총 4368개
    # words_dictionary = {words_list[i]: words_freq[i] for i in range(len(words_freq))}
    # print(words_dictionary)

    
    logger.info(f'Creating vocabulary: {len(words_list)} tokens!')
    write_pickle_file(words_list, 'data/{}/pickles/words_list.p'.format(dataset))


def pack_augment_dataset_to_hdf5(dataset):

    splits = ['train', 'val', 'test']
    sampling_rate = 32000
    all_captions = [] 
    #csv file에 있는 caption들을 list로 한번에 저장 ex) train data -> 3839*5 = 19195개 caption 저장
    
    #if dataset == 'AudioCaps':
    #    audio_duration = 10
    if dataset == 'Clotho':
        audio_duration = 30
        
    else:
        raise NotImplementedError(f'No dataset named: {dataset}')

    max_audio_length = audio_duration * sampling_rate # 30 * 32000
    
    
    for split in splits:
        csv_path = 'data/{}/csv_files/{}.csv'.format(dataset, split)
        audio_dir = 'data/{}/waveforms/{}/'.format(dataset, split)
        hdf5_path = 'data/{}/hdf5s/{}/'.format(dataset, split)

        # make dir for hdf5
        Path(hdf5_path).mkdir(parents=True, exist_ok=True)

        meta_dict = load_metadata(dataset, csv_path)
        # meta_dict: {'audio_names': [], 'captions': []}

        audio_nums = len(meta_dict['audio_name'])

        if split == 'train':
            # store all captions in training set into a list
            if dataset == 'Clotho':
                for caps in meta_dict['captions']:
                    for cap in caps:
                        all_captions.append(cap)
            else:
                all_captions.extend(meta_dict['captions'])

        start_time = time.time()

        ### file name = train_augment.h5
        # augment는 train set에서만 적용시켜야하니까 코드 나눠서 만들기 


        if split=='train':
            with h5py.File(hdf5_path+'{}_augment.h5'.format(split), 'w') as hf:
                hf.create_dataset('audio_name', shape=(audio_nums,), dtype=h5py.special_dtype(vlen=str))
                hf.create_dataset('audio_length', shape=(audio_nums,), dtype=np.uint32)
                hf.create_dataset('waveform', shape=(audio_nums, max_audio_length), dtype=np.float32)

                if split == 'train' and dataset == 'AudioCaps':
                    hf.create_dataset('caption', shape=(audio_nums,), dtype=h5py.special_dtype(vlen=str))
                
                else: #'clotho dataset'
                    hf.create_dataset('caption', shape=(audio_nums, 5), dtype=h5py.special_dtype(vlen=str))

                for i in tqdm(range(audio_nums)):
                    audio_name = meta_dict['audio_name'][i]

                    audio, _ = librosa.load(audio_dir + audio_name, sr=sampling_rate, mono=True)
                    audio, audio_length = pad_or_truncate(audio, max_audio_length)

                    #augmentation




                    hf['audio_name'][i] = audio_name.encode()
                    hf['audio_length'][i] = audio_length
                    hf['waveform'][i] = audio
                    hf['caption'][i] = meta_dict['captions'][i]


        else:
            with h5py.File(hdf5_path+'{}_augment.h5'.format(split), 'w') as hf:
                # file name contains 'augment' but does not implement augmentation b/c these are not the training data.
                # just for separating h5 file name with not augmented file.

                hf.create_dataset('audio_name', shape=(audio_nums,), dtype=h5py.special_dtype(vlen=str))
                hf.create_dataset('audio_length', shape=(audio_nums,), dtype=np.uint32)
                hf.create_dataset('waveform', shape=(audio_nums, max_audio_length), dtype=np.float32)

                if split == 'train' and dataset == 'AudioCaps':
                    hf.create_dataset('caption', shape=(audio_nums,), dtype=h5py.special_dtype(vlen=str))
                
                else: #'clotho dataset'
                    hf.create_dataset('caption', shape=(audio_nums, 5), dtype=h5py.special_dtype(vlen=str))

                for i in tqdm(range(audio_nums)):
                    audio_name = meta_dict['audio_name'][i]

                    audio, _ = librosa.load(audio_dir + audio_name, sr=sampling_rate, mono=True)
                    audio, audio_length = pad_or_truncate(audio, max_audio_length)


                    hf['audio_name'][i] = audio_name.encode()
                    hf['audio_length'][i] = audio_length
                    hf['waveform'][i] = audio
                    hf['caption'][i] = meta_dict['captions'][i]

        logger.info(f'Packed {split} set to {hdf5_path} using {time.time() - start_time} s.')
    
    words_list, words_freq = _create_vocabulary(all_captions)
    # 총 4368개
    # words_dictionary = {words_list[i]: words_freq[i] for i in range(len(words_freq))}
    # print(words_dictionary)

    
    logger.info(f'Creating vocabulary: {len(words_list)} tokens!')
    write_pickle_file(words_list, 'data/{}/pickles/augment_words_list.p'.format(dataset))


def _create_vocabulary(captions):
    
    vocabulary = []
    for caption in captions:
        caption_words = caption.strip().split()
        vocabulary.extend(caption_words)
    words_list = list(set(vocabulary))
    words_list.sort(key=vocabulary.index)
    words_freq = [vocabulary.count(word) for word in words_list]
    words_list.append('<sos>')
    words_list.append('<eos>')
    words_list.append('<ukn>')
    words_freq.append(len(captions))
    words_freq.append(len(captions))
    words_freq.append(0)

    return words_list, words_freq


def _sentence_process(sentence, add_specials=False):

    # transform to lower case
    sentence = sentence.lower()

    if add_specials:
        sentence = '<sos> {} <eos>'.format(sentence)

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    return sentence


def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    length = len(x)
    if length <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - length)), axis=0), length
    else:
        return x[:audio_length], audio_length

