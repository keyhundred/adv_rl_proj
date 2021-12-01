import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F

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
    
    temp_clean1 = temp_clean[:,0:200]
    temp_noisy1 = temp_noisy[:,0:200]
    
    temp_clean2 = temp_clean[:,200:400]
    temp_noisy2 = temp_noisy[:,200:400]
    
    temp_clean3 = temp_clean[:,400:600]
    temp_noisy3 = temp_noisy[:,400:600]
    
    temp_clean4 = temp_clean[:,600:800]
    temp_noisy4 = temp_noisy[:,600:800]
    
    temp_clean5 = temp_clean[:,800:1000]
    temp_noisy5 = temp_noisy[:,800:1000]
    
    temp_clean1 = temp_clean1[None]
    temp_noisy1 = temp_noisy1[None]
    
    temp_clean2 = temp_clean2[None]
    temp_noisy2 = temp_noisy2[None]
    
    temp_clean3 = temp_clean3[None]
    temp_noisy3 = temp_noisy3[None]
    
    temp_clean4 = temp_clean4[None]
    temp_noisy4 = temp_noisy4[None]
    
    temp_clean5 = temp_clean5[None]
    temp_noisy5 = temp_noisy5[None]
    
    clean_cat = torch.cat([torch.tensor(temp_clean1),torch.tensor(temp_clean2),\
                           torch.tensor(temp_clean3),torch.tensor(temp_clean4),\
                          torch.tensor(temp_clean5)], dim = 0)
    
    noisy_cat = torch.cat([torch.tensor(temp_noisy1),torch.tensor(temp_noisy2),\
                           torch.tensor(temp_noisy3),torch.tensor(temp_noisy4),\
                          torch.tensor(temp_noisy5)], dim = 0)
    
    clean_data.append(clean_cat)
    noisy_data.append(noisy_cat)

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