import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ppo_filt import PPO
# from preproc import train_dataset, train_dataloader, val_dataloader, val_dataset, test_dataloader, test_dataset
from preproc_mel import train_dataset
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

action = [
    -5,
    -2,
    -1,
    -0.1,
    -0.001,
    -0.0001,
    0,
    0.0001,
    0.001,
    0.1,
    1,
    2,
    5,
]

def get_action(a):
    '''
    input: Categorical().sample().numpy()
    output: np.ndarray (1, 128, 100)
    '''
    action_list = []
    for i in range(a.shape[1]):
        tmp = []
        for j in range(a.shape[2]):
            tmp.append(action[a[0][i][j]])
        action_list.append(tmp)

    return np.array(action_list)[None]

model = PPO()
model = torch.load('ppo_filt_600_-0.4735.pth')

N_TEST = len(train_dataset)
# N_TEST = 10
START_IDX = 0
total_sound = None
for i in range(START_IDX, START_IDX + N_TEST):
    dat = train_dataset[i][0][None]
    clean = train_dataset[i][1].numpy()[0]

    prob = model.pi(dat.float()).permute((0, 2, 3, 1))
    m = Categorical(prob)
    a = m.sample().numpy()
    a = get_action(a)[0]

    s = dat.numpy()[0, 0]
    sp = s + a
    
    if total_sound is None:
        total_sound = sp
    else:
        total_sound = np.append(total_sound, sp, axis=-1)

    # plt.subplot(141)
    # plt.imshow(s)
    # # plt.colorbar()

    # plt.subplot(142)
    # plt.imshow(a)
    # # plt.colorbar()

    # plt.subplot(143)
    # plt.imshow(sp)
    # # plt.colorbar()

    # plt.subplot(144)
    # plt.imshow(clean)
    # # plt.colorbar()

    # plt.show()
    # plt.clf()

print(total_sound.shape)
S = librosa.feature.inverse.mel_to_stft(total_sound)
y = librosa.griffinlim(S, win_length=320, hop_length=160)

sf.write("result.wav", y, samplerate=16000, endian='LITTLE', subtype='PCM_16')