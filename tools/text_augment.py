import time
import os
from itertools import chain
import random
import math

import torch
import torch.nn as nn
import librosa

import numpy as np
from re import sub
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from tools.file_io import load_csv_file, write_pickle_file
from tools.dataset import _create_vocabulary, load_metadata

from nltk.corpus import wordnet 
import re

import random
from random import shuffle
random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
              'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
              'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
              'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
              'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
              'have', 'has', 'had', 'having', 'do', 'does', 'did','doing', 'a', 'an', 
              'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
              'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
              'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
              'in','out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
              'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
              'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
              'don', 'should', 'now', '']


def get_only_chars(line):
    """
    #cleaning up text
    """

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)
    
    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1

    #Synonym replacement(SR)
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr*num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    #Random insertion(RI)
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

    #Random swap(RS)
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

    #Random deletion(RD)
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    #trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    #append the original sentence
    augmented_sentences.append(sentence)

    return augmented_sentences


########################################################################
# Synonym replacement(SR)
# Replace n words in the sentence with synonyms from wordnet
########################################################################

def get_synonyms(word):
    """
    Get synonyms of a word
    """
    synonyms = set()
    
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)



def synonym_replacement(words, n):
    
    words = words.split()
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence


########################################################################
# Random insertion(RI)
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

    

########################################################################
# Random deletion(RD)
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap(RS)
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    """
    random.randint(0,x): 0<=len<=x 사이의 랜덤한 정수를 반환
    """
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words



##########################################
#수정해야함
def pack_text_augment_dataset_to_hdf5(dataset):
    """

    Args:
        dataset:'Clotho'

    Returns:

    """

    splits = ['train_augment', 'val', 'test']
    sampling_rate = 32000
    all_captions = [] 
    #csv file에 있는 caption들을 list로 한번에 저장 ex) train data -> 3839*5 = 19195개 caption 저장
    
    if dataset == 'Clotho':
        audio_duration = 30
        
    else:
        raise NotImplementedError(f'No dataset named: {dataset}')

    max_audio_length = audio_duration * sampling_rate # 30 * 32000
    
    for split in splits:
        csv_path = 'data/{}/csv_files/{}.csv'.format(dataset, split)
        audio_dir = 'data/{}/waveforms/{}/'.format(dataset, split)
        
        if split=='train_augment':
            hdf5_path = 'data/{}/hdf5s/{}/'.format(dataset, split)
              
        else:
            hdf5_path = 'data/{}/hdf5s/{}_augment/'.format(dataset, split)

            
        # make dir for hdf5
        Path(hdf5_path).mkdir(parents=True, exist_ok=True)

        meta_dict = load_metadata(dataset, csv_path)
        # meta_dict: {'audio_names': [], 'captions': []}

        audio_nums = len(meta_dict['audio_name'])

        if split == 'train_augment':
            # store all captions in training set into a list
            if dataset == 'Clotho':
                for caps in meta_dict['captions']:
                    for cap in caps:
                        all_captions.append(cap)
            else:
                all_captions.extend(meta_dict['captions'])

        start_time = time.time()
        
        
        #h5py 파일 생성
        if split=='train_augment':
            
            try:
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
            
            except FileNotFoundError as F:
                print("file not found", F)
                
            except ValueError:
                print("Invalid value.")
                
            except Exception as e:
                print(f"Error: {e}")
                

            logger.info(f'Packed {split} set to {hdf5_path} using {time.time() - start_time} s.')
        
    words_list, words_freq = _create_vocabulary(all_captions)
    # 총 4368개(augment전)
    
    logger.info(f'Creating vocabulary: {len(words_list)} tokens!')
    write_pickle_file(words_list, 'data/{}/pickles/words_list_text_augment.p'.format(dataset))
