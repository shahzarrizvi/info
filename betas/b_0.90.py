# Imports.
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

import os
import tqdm as tqdm

import numpy as np
from scipy import stats

# Utility functions.
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

def train_flow(flow, optimizer, data, dir, num_iter = 40000):
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

def marginal_X(X, log_prob):
    X = X.repeat_interleave(101).reshape(-1, 1)
    
    Y = torch.linspace(-0.5, 1.5, 101).reshape(-1, 1)
    Y = Y.repeat(n, 1)
    Y = Y.to(device)

    XY = torch.cat([X, Y], dim = 1)
    LP = log_prob(XY)
    PX = []
    for i in range(n):
        PX += [torch.trapezoid(torch.exp(LP[101 * i : 101 * (i + 1)]), dx = 0.02)]
    return torch.stack(PX, dim = 0)

def marginal_Y(Y, log_prob):
    Y = Y.repeat_interleave(101).reshape(-1, 1)
    
    X = torch.linspace(-0.5, 1.5, 101).reshape(-1, 1)
    X = X.repeat(n, 1)
    X = X.to(device)

    XY = torch.cat([X, Y], dim = 1)
    LP = log_prob(XY)
    PY = []
    for i in range(n):
        PY += [torch.trapezoid(torch.exp(LP[101 * i : 101 * (i + 1)]), dx = 0.02)]
    return torch.stack(PY, dim = 0)

def s(X, Y, D, flow):
    LPX = torch.log(marginal_X(X, flow.log_prob))
    LPY = torch.log(marginal_Y(Y, flow.log_prob))
    return flow.log_prob(D).mean() - LPX.mean() - LPY.mean() 

## Assign the parameters for this experiment.
np.random.seed(666)
n = 2000
α = np.pi/4
b = 0.9
a = 1
dir = 'b/{}/'.format(b)

U = stats.uniform.rvs(scale = 2*np.pi, size = n)
R = np.array([[np.cos(α), -np.sin(α)], [np.sin(α), np.cos(α)]])

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

# Shuffle the data.
idx = np.random.choice(n, size = n, replace = False)
X_star = D[idx, 0].reshape(-1, 1)
Y_star = Y
D_star = torch.cat([X_star, Y_star], dim = 1)

# Train the joint flow.
flow = make_flow(d = 2)
flow.to(device)
optimizer = optim.Adam(flow.parameters())

os.makedirs(dir + 'flow', exist_ok = True)
train_flow(flow, optimizer, D, dir + 'flow/')

# Train the null flow.
flow_0 = make_flow(d = 2)
flow_0.to(device)
optimizer_0 = optim.Adam(flow_0.parameters())

os.makedirs(dir + 'flow_0', exist_ok = True)
train_flow(flow_0, optimizer_0, D_star, dir + 'flow/')

# Compute the observed coefficient.
s_obs = s(X, Y, D, flow).cpu().detach().numpy()
np.save(dir + 's_obs.npy'.format(n), s_obs)
print(s_obs)

# Sample from null distribution via the bootstrap.
reps = 1000
ss = np.zeros(reps)
for i in range(reps):
    # Resample from D_star.
    idx = np.random.choice(n, size = n)
    D_boot = D_star[idx]
    X_boot = D_boot[:, 0].reshape(-1, 1)
    Y_boot = D_boot[:, 1].reshape(-1, 1)

    # Compute the information coefficient.
    ss[i] = s(X_boot, Y_boot, D_boot, flow_0)
    print(i, '\t', ss[i])
np.save(dir + 'ss.npy'.format(n), ss)

# Compute the p value.
s_p_val = np.mean(ss >= s_obs)
np.save(dir + 's_p_val.npy'.format(n), s_p_val)
print(s_p_val)