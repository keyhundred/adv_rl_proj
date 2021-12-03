import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from preproc_mel_res import train_dataset
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
# Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.n_action = len(action)

        self.latent = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.Conv2d(16, 16, 3, padding=1),
        )
        self.fc_pi = nn.Conv2d(16, 11, 3, padding=1)
        self.fc_v  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*128*100, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 1):
        x = F.relu(self.latent(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.latent(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s[0].numpy())
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime[0].numpy())
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = (r + gamma * self.v(s_prime) * done_mask).float()
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1).permute((0, 2, 3, 1))
            pi_a = get_prob(pi[0][None], a[0])[None]
            for i in range(1, pi.shape[0]):
                pi_tmp = get_prob(pi[i][None], a[i])[None]
                pi_a = torch.cat([pi_a, pi_tmp], dim=-1)
            pi_a = pi_a.unsqueeze(dim=-1)
            ratio = torch.exp(pi_a - prob_a)  # a/b == exp(log(a)-log(b))
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

model = PPO()
model = torch.load('ppo_filt_model2/ppo_filt_80_-17.7351.pth')

# N_TEST = len(train_dataset)
N_TEST = 10
START_IDX = len(train_dataset) - 10
total_sp = None
total_n = None
for i in range(START_IDX, START_IDX + N_TEST):
    dat = train_dataset[i][0][None]
    clean = train_dataset[i][1].numpy()[0]

    prob = model.pi(dat.float()).permute((0, 2, 3, 1))
    m = Categorical(prob)
    a = m.sample().numpy()
    a = get_action(a)[0]

    s = dat.numpy()[0, 0]
    sp = s + a
    
    if total_sp is None:
        total_sp = sp
    else:
        total_sp = np.append(total_sp, sp, axis=-1)
        
    if total_n is None:
        total_n = clean
    else:
        total_n = np.append(total_n, clean, axis=-1)

    # plt.subplot(221)
    # plt.imshow(s)
    # plt.colorbar()

    # plt.subplot(222)
    # plt.imshow(a)
    # plt.colorbar()

    # plt.subplot(223)
    # plt.imshow(sp)
    # plt.colorbar()

    # plt.subplot(224)
    # plt.imshow(clean)
    # plt.colorbar()

    # plt.show()
    # plt.clf()

print(total_sp.shape)
print(total_n.shape)

S = librosa.feature.inverse.mel_to_stft(total_sp)
y = librosa.griffinlim(S, win_length=320, hop_length=160)

print('denoised ready')
sf.write("result.wav", y, samplerate=16000, endian='LITTLE', subtype='PCM_16')

print('denoised')

S = librosa.feature.inverse.mel_to_stft(total_n)
y = librosa.griffinlim(S, win_length=320, hop_length=160)

sf.write("result_noise.wav", y, samplerate=16000, endian='LITTLE', subtype='PCM_16')
print('noise')