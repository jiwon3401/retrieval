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
from loguru import logger
from tools.dataset import load_metadata, _create_vocabulary, _sentence_process, pad_or_truncate


    
def pack_augment_dataset_to_hdf5(dataset):
    """

    Args:
        dataset:'Clotho'

    Returns:

    """

    splits = ['train_augment', 'val', 'test']
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
    write_pickle_file(words_list, 'data/{}/pickles/words_list_augment.p'.format(dataset))

    

if __name__ == '__main__':
    #logger.info('Creating Clotho augmentation files...')
    #augment_raw_audio(dataset, train_meta_dict)
    #logger.info('Creating raw audio DONE!')
    
    logger.info('Packing Augmented Clotho data...')
    pack_augment_dataset_to_hdf5('Clotho')
    logger.info('augmented Clotho done!')
    
    
    