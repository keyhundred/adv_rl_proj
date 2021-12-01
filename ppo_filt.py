import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from preproc import train_dataset, train_dataloader, val_dataloader, val_dataset, test_dataloader, test_dataset
from denoise_model import unet_model, ae_model

filter_size = [3, 3]
data_size = [1000, 60]
filter = [
    
]


#Hyperparameters
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
        self.n_action = len(filter)
        
        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,self.n_action)
        self.fc_v  = nn.Linear(256,1)

        self.fc_inv = nn.Linear(256*2, self.n_action)
        self.ce = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def inv(self, state, next_state):
        s = self.fc1(state)
        sp = self.fc1(next_state)
        joint_state = torch.cat([s, sp], dim=1)
        pred_action = torch.relu(self.fc_inv(joint_state))
        return pred_action
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
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

        USE_INV = False
        INV_COEF = 0.1

        for i in range(K_epoch):
            if USE_INV:
                pred_action = self.inv(s, s_prime)
                inv_loss = self.ce(pred_action, a.reshape((-1,))).mean()

            td_target = r + gamma * self.v(s_prime) * done_mask
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
        
def make_filter(ones, filt, x, y):
    for i in range(filter_size[0]):
        for j in range(filter_size[1]):
            ones[x + i][y + j] = filt[i][j]
    return ones
    
def main():
    # Dataset
    original_dataset = np.random.randn(300, 1000, 60)
    noised_dataset = np.random.randn(300, 1000, 60)

    done_repeat = 20
    len_data = original_dataset.size[0]

    model = PPO()
    score = 0.0
    print_interval = 20

    idx = 0
    for n_epi in range(10000):
        done = False
        x = 0
        y = 0
        filtered = 0
        s = noised_dataset[idx]
        label = noised_dataset[idx]

        while not done:
            for t in range(T_horizon):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                selected_filter = filter[a]

                # s_prime, r, done = step(a)
                filt = make_filter(np.ones(data_size), selected_filter, x, y)
                s_prime = s * filt
                r = 0

                if x >= data_size[0] - filter_size[0] and y >= data_size[1] - filter_size[1] :
                    r = -1 * np.mean(np.abs(label - s_prime))
                    filtered += 1
                    if filtered >= done_repeat:
                        filtered = 0
                        done = True
                    else:
                        done = False
                        x = 0
                        y = 0
                else:
                    x = (x + 1) % (data_size[0] - filter_size[0])
                    y = (y + 1) % (data_size[1] - filter_size[1])


                model.put_data((s, a, r, s_prime, prob[a].item(), done))
                s = s_prime

                if done:
                    idx = (idx + 1) % len_data
                    s = original_dataset[idx]

                score += r
                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0


if __name__ == '__main__':
    main()