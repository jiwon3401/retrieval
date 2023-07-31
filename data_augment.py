from tools.dataset import pack_augment_dataset_to_hdf5
from tools.audio_augment import augment_raw_audio
from loguru import logger

if __name__ == '__main__':
    #logger.info('Creating Clotho augmentation files...')
    #augment_raw_audio(dataset, train_meta_dict)
    #logger.info('Creating raw audio DONE!')
    
    logger.info('Packing Augmented Clotho data...')
    pack_augment_dataset_to_hdf5('Clotho')
    logger.info('augmented Clotho done!')