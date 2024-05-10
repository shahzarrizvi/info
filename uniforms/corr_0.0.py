# Imports.
import numpy as np
from scipy import stats

def corr(X, Y):
    return np.corrcoef(X, Y)[0][1]

## Assign the parameters for this experiment.
np.random.seed(666)
n = 2000
r = 0.0
dir = 'r/{}/'.format(r)

# Generate the true data.
C = stats.multivariate_normal.rvs(np.zeros(2), [[1, r],[r, 1]], size = 10000)
C = C[:n]
D = stats.norm.cdf(C)
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
print(r_obs)

# Sample from null distribution via the bootstrap.
reps = 10000
rs = np.zeros(reps)
for i in range(reps):
    # Resample from D_star.
    idx = np.random.choice(n, size = n)
    D_boot = D_star[idx]
    X_boot = D_boot[:, 0]
    Y_boot = D_boot[:, 1]

    # Compute the information coefficient.
    rs[i] = corr(X_boot, Y_boot)
    print(i, '\t', rs[i])
np.save(dir + 'rs.npy', rs)

# Compute the p value.
r_p_val = np.mean(abs(rs) >= abs(r_obs))
np.save(dir + 'r_p_val.npy', r_p_val)
print(r_p_val)