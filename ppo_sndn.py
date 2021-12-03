from math import fabs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from preproc import train_dataset, train_dataloader, val_dataloader, val_dataset, test_dataloader, test_dataset
from denoise_model import unet_model, ae_model, cl_model

filter = [
    unet_model, ae_model, cl_model
]


#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 5

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.n_action = len(filter)
        
        self.latent= nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=2),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.Conv2d(64, 5, 3, stride=2),
            nn.Flatten(),
        )
        self.fc_pi = nn.Linear(3465, self.n_action)
        self.fc_v  = nn.Linear(3465, 1)

        self.fc_inv = nn.Linear(3465*2, self.n_action)
        self.ce = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.latent(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.latent(x))
        v = self.fc_v(x)
        return v

    def inv(self, state, next_state):
        s = self.latent(state)
        sp = self.latent(next_state)
        joint_state = torch.cat([s, sp], dim=1)
        pred_action = torch.relu(self.fc_inv(joint_state))
        return pred_action
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s.squeeze().numpy())
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime.squeeze().numpy())
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        # print(s_lst)
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        USE_INV = True
        INV_COEF = 0.1

        for i in range(K_epoch):
            if USE_INV:
                pred_action = self.inv(s, s_prime)
                inv_loss = self.ce(pred_action, a.reshape((-1,))).mean()

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

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            if USE_INV:
                loss += inv_loss * INV_COEF

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def step(s, a, clean, time_step):
    s_prime = filter[a].forward(s).detach()

    time_step += 1
    if time_step >= 5:
        time_step = 0
        r = -1 * np.sqrt(np.mean(((s_prime-clean)**2).numpy(), keepdims=False))
        done = True
    else:
        r = 0
        done = False

    return s_prime, r, done, time_step
    
    
def main():
    model = PPO()
    score = 0.0
    print_interval = 1

    time_step = 0
    idx = 0
    s = train_dataset[idx][0].unsqueeze(dim=0)
    clean = train_dataset[idx][1].unsqueeze(dim=0)
    for n_epi in range(10000):
        done = False
        while not done:
            for t in range(T_horizon):
                prob = model.pi(s)[0]
                m = Categorical(prob)
                a = m.sample().item()

                s_prime, r, done, time_step = step(s, a, clean, time_step)

                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    idx = (idx + 1) % len(train_dataset)
                    s = train_dataset[idx][0].unsqueeze(dim=0)
                    clean = train_dataset[idx][1].unsqueeze(dim=0)
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            torch.save(model.state_dict(), 'models/ppo_select_filter_{}_{:.4f}'.format(n_epi, score/print_interval))
            score = 0.0


if __name__ == '__main__':
    main()