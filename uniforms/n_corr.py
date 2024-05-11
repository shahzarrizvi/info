# Imports.
import os
import numpy as np
from scipy import stats

def corr(X, Y):
    return np.corrcoef(X, Y)[0][1]

## Assign the parameters for this experiment.
np.random.seed(666)
r = 0.01

for n in [100, 500, 1000, 2000, 5000]:
    dir = 'n/{}/'.format(n)
    os.makedirs(dir, exist_ok = True)
    
    # Generate the true data.
    C = stats.multivariate_normal.rvs(np.zeros(2), [[1, r],[r, 1]], size = 10000)
    C = C[:n]
    D = stats.norm.cdf(C)
    X = D[:, 0]
    Y = D[:, 1]
    
    # Compute the observed coefficient.
    r_obs = corr(X, Y)
    np.save(dir + 'r_obs.npy', r_obs)
    print(n, '\t', r_obs, end = '\t')
    
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
    r_p_val = np.mean(abs(rs) >= abs(r_obs))
    np.save(dir + 'r_p_val.npy', r_p_val)
    print(r_p_val)