import numpy as np
import pickle
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from policy import GaussianPolicy
from rl_games.algos_torch.running_mean_std import RunningMeanStd

import os
with open(file=os.path.join(os.path.abspath(os.path.dirname(__file__)),'../data/dataset.pkl'), mode='rb') as f:
    dataset = pickle.load(f)

action_batch = torch.Tensor(dataset['action'])
qpos_batch = torch.Tensor(dataset['qpos'])
qvel_batch = torch.Tensor(dataset['qvel'])
obs_batch = torch.cat((qpos_batch, qvel_batch), dim=1)

print("action : ", action_batch.shape)
print("qpos : ", qpos_batch.shape)
print("qvel : ", qvel_batch.shape)

obs_dim = obs_batch.shape[1]
action_dim = action_batch.shape[1]
hidden_dim = 512

del dataset

policy = GaussianPolicy(
    input_dim=obs_dim,
    output_dim=action_dim,
    hidden_dim=hidden_dim,
    is_deterministic=False,
)

class MPCDataset(Dataset):
    def __init__(self, obs, act):
        self.obs = obs
        self.act = act
        assert self.obs.shape[0] == self.act.shape[0]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self,idx):
        return self.obs[idx], self.act[idx]
    
train_dataset = MPCDataset(obs_batch, action_batch)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

num_epoch = 100
optimizer = optim.Adam(policy.parameters(), lr=3e-4)
loss = torch.nn.MSELoss()
def criterion(output: torch.tensor, y: torch.tensor):
    return loss(output, y)

for epoch in range(num_epoch):

    with tqdm(train_dataloader, unit="batch") as tepoch:
        
        for x, y in tepoch:
            
            tepoch.set_description(f"Epoch {epoch+1}")

            optimizer.zero_grad()
                
            output, _ = policy(x)
            l = criterion(output, y)
            l.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss=l.item())
