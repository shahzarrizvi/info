import torch
from torch import nn
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation

import tqdm as tqdm
import numpy as np
from scipy import stats
import os

def make_checkpoint(flow, optimizer, loss, filename):
    torch.save({'model_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
               }, 
               filename)

def make_flow(d, num_layers = 5):
    base_dist = StandardNormal(shape=[d])
    
    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=d))
        transforms.append(MaskedAffineAutoregressiveTransform(features=d, 
                                                              hidden_features=8))
    transform = CompositeTransform(transforms)
    
    return Flow(transform, base_dist)

def train_flow(flow, optimizer, data, dir, num_iter = 10000):
    losses = np.zeros(num_iter)
    print(-flow.log_prob(inputs=data).mean())
    for i in tqdm.trange(num_iter):
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=data).mean()
        losses[i] = loss
        
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            np.save(dir + 'losses.npy', losses)
            make_checkpoint(flow, optimizer, loss, dir + 'ckpt_{}'.format(i))
            
    make_checkpoint(flow, optimizer, loss, dir + 'ckpt'.format(num_iter))
    np.save(dir + 'losses.npy', losses)

## Data parameters
np.random.seed(666)
U = stats.uniform.rvs(scale = 2*np.pi, size = n)
R = np.array([[np.cos(α), -np.sin(α)], [np.sin(α), np.cos(α)]])

α = np.pi/4
n = 10000
bb = np.round(np.linspace(0, 1, 11), 1)
for b in bb:
    # Create data
    a = np.sqrt(2 - a**2)
    V = np.vstack([a * np.cos(U), b * np.sin(U)])
    D = (R @ V).T
    X = D[:, 0].reshape(n, 1)
    Y = D[:, 1].reshape(n, 1)

    D = torch.tensor(D, dtype = torch.float32).cuda()
    X = torch.tensor(X, dtype = torch.float32).cuda()
    Y = torch.tensor(Y, dtype = torch.float32).cuda()
    D.to(device)
    X.to(device)
    Y.to(device)

    # Train XY flow
    flow_XY = make_flow(d = 2)
    flow_XY.to(device)
    optimizer_XY = optim.Adam(flow_XY.parameters())

    os.makedirs('b/{}/XY'.format(b), exist_ok = True)
    train_flow(flow_XY, optimizer_XY, D, 'b/{}/XY/'.format(b))
    
    # Train X flow
    flow_X = make_flow(d = 1)
    flow_X.to(device)
    optimizer_X = optim.Adam(flow_X.parameters())
    os.makedirs('b/{}/X'.format(b), exist_ok = True)
    train_flow(flow_X, optimizer_X, D, 'b/{}/X/'.format(b))
    
    # Train Y flow
    flow_Y = make_flow(d = 1)
    flow_Y.to(device)
    optimizer_Y = optim.Adam(flow_.parameters())
    os.makedirs('b/{}/Y'.format(b), exist_ok = True)
    train_flow(flow_Y, optimizer_Y, D, 'b/{}/Y/'.format(b))