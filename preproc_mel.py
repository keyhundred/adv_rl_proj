import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import librosa

class speech_dataset(Dataset):

    def __init__(self, clean, noisy):
        self.clean = clean
        self.noisy = noisy

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):

        clean_data = self.clean[idx]
        noisy_data = self.noisy[idx]


        out = [
            clean_data,
            noisy_data
        ]

        return out


clean_list_name = os.listdir('stft2/clean') 
noisy_list_name = os.listdir('stft2/noisy')

clean_data = []
noisy_data = []


for i in tqdm(range(len(clean_list_name))):
    temp_clean = np.load('stft2/clean/' + clean_list_name[i])
    temp_noisy = np.load('stft2/noisy/' + noisy_list_name[i])
    
    temp_clean = temp_clean[:,0:1000]
    temp_noisy = temp_noisy[:,0:1000]

    temp_clean = librosa.feature.melspectrogram(S=temp_clean**2, sr=16000)
    temp_noisy = librosa.feature.melspectrogram(S=temp_noisy**2, sr=16000)

    temp_clean = torch.Tensor(temp_clean[None])
    temp_noisy = torch.Tensor(temp_noisy[None])

    clean_data.append(temp_clean)
    noisy_data.append(temp_noisy)

BATCH_SIZE = 4

train_dataset = speech_dataset(clean = clean_data[:int(len(clean_data)*0.7)],  \
                                    noisy=noisy_data[:int(len(clean_data)*0.7)])

train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, sampler = RandomSampler(train_dataset), num_workers = 0)

val_dataset = speech_dataset(clean = clean_data[int(len(clean_data)*0.7):int(len(clean_data)*0.85)],  \
                                    noisy=noisy_data[int(len(clean_data)*0.7):int(len(clean_data)*0.85)])

val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, sampler = SequentialSampler(val_dataset), num_workers = 0)

test_dataset = speech_dataset(clean = clean_data[int(len(clean_data)*0.85):],  \
                                    noisy=noisy_data[int(len(clean_data)*0.85):])
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, sampler = SequentialSampler(test_dataset), num_workers = 0)