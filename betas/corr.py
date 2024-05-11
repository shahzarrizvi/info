# Imports.
import os
import numpy as np
from scipy import stats

def corr(X, Y):
    return np.corrcoef(X, Y)[0][1]

## Assign the parameters for this experiment.
np.random.seed(666)
n = 2000

α = np.pi/4
R = np.array([[np.cos(α), -np.sin(α)], [np.sin(α), np.cos(α)]])

a = 1
for b in np.round(np.linspace(0.9, 1, 11), 2):
    dir = 'b/{}/'.format(b)
    os.makedirs(dir, exist_ok = True)
    
    # Generate the true data.
    U = stats.uniform.rvs(scale = 2*np.pi, size = n)
    V = np.vstack([a * np.cos(U), b * np.sin(U)])
    D = (R @ V).T
    X = D[:, 0]
    Y = D[:, 1]
    
    # Shuffle the data.
    idx = np.random.choice(n, size = n, replace = False)
    X_star = D[idx, 0]
    Y_star = Y
    D_star = np.hstack([X_star.reshape(-1, 1), Y_star.reshape(-1, 1)])
    
    # Compute the observed coefficient.
    r_obs = corr(X, Y)
    np.save(dir + 'r_obs.npy', r_obs)
    print(b, '\t', r_obs, end = '\t')
    
    # Sample from null distribution via the bootstrap.
    reps = 5000
    rs = np.zeros(reps)
    for i in range(reps):
        # Shuffle the data.
        idx = np.random.choice(n, size = n, replace = False)
        X_boot = D[idx, 0]
        Y_boot = Y
    
        # Compute the information coefficient.
        rs[i] = corr(X_boot, Y_boot)
        #print(i, '\t', rs[i])
    np.save(dir + 'rs.npy', rs)
    
    # Compute the p value.
    r_p_val = np.mean(rs >= r_obs)
    np.save(dir + 'r_p_val.npy', r_p_val)
    print(r_p_val)