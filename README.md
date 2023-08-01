# Audio-Text Retrieval

### Text-to-Audio Retrieval code repository
* Audio-text matching

* This task is concerned with retrieving audio signals using their sound content textual descriptions (i.e., audio captions). Human written audio captions will be used as text queries. For each text query, the goal of this task is to retrieve 10 audio files from a given dataset and sort them based their match with the query. Through this subtask, we aim to inspire further research into language-based audio retrieval with unconstrained textual descriptions.


### File Structure

```
  Audio Retrieval
  ├── data
  │   ├── Clotho  
  │       ├── waveforms
  │       ├── csv_files
  │       ├── hdf5s
  │       ├── pickles
  │
  │   ├── musan  
  │
  ├── data_handling
  │   ├── DataLoader.py  
  │      
  ├── models
  │   ├── ASE_model.py  
  │   ├── AudioEncoder.py  
  │   ├── TextEncoder.py  
  │      
  ├── models
  │   ├── ASE_model.py  
  │   
  ├── pretrained_models
  │    
  ├── tools
  │   ├── config_loader.py  
  │   ├── dataset.py  
  │   ├── file_io.py  
  │   ├── loss.py    
  │   ├── utils.py  
  │   ├── audio_augment.py    
  │   ├── text_augment.py    
  │
  ├── trainer
  │   ├── trainer.py  
  │  
  ├── train.py
  
  ```