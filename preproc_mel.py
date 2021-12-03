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


for i in tqdm(range(int(len(clean_list_name) * 0.1))):
    temp_clean = np.load('stft2/clean/' + clean_list_name[i])
    temp_noisy = np.load('stft2/noisy/' + noisy_list_name[i])

    for j in range(0, temp_noisy.shape[1], 100):

        if temp_noisy.shape[1] <= j + 100:
            break

        dat_clean = temp_clean[:, j:j+100]
        dat_noisy = temp_noisy[:, j:j+100]

        dat_clean = librosa.feature.melspectrogram(S=dat_clean**2, sr=16000)
        dat_noisy = librosa.feature.melspectrogram(S=dat_noisy**2, sr=16000)

        dat_clean = torch.Tensor(dat_clean[None])
        dat_noisy = torch.Tensor(dat_noisy[None])

        clean_data.append(dat_clean)
        noisy_data.append(dat_noisy)

BATCH_SIZE = 4

print(f"train data : {len(clean_data)}")
train_dataset = speech_dataset(clean = clean_data,  \
                                    noisy=noisy_data)

# train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, sampler = RandomSampler(train_dataset), num_workers = 0)

# val_dataset = speech_dataset(clean = clean_data[int(len(clean_data)*0.7):int(len(clean_data)*0.85)],  \
#                                     noisy=noisy_data[int(len(clean_data)*0.7):int(len(clean_data)*0.85)])

# val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, sampler = SequentialSampler(val_dataset), num_workers = 0)

# test_dataset = speech_dataset(clean = clean_data[int(len(clean_data)*0.85):],  \
#                                     noisy=noisy_data[int(len(clean_data)*0.85):])
# test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, sampler = SequentialSampler(test_dataset), num_workers = 0)
