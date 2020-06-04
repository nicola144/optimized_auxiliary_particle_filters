
import numpy as np 
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import sys

x, y = np.mgrid[-1:1:.01, -1:1:.01]

pos = np.empty(x.shape + (2,))

print(pos.shape)
sys.exit()
pos[:, :, 0] = x 
pos[:, :, 1] = y

rv = multivariate_normal(mean=[0.5, -0.2], cov=[[2.0, 0.3], [0.3, 0.5]])

plt.contourf(x, y, rv.pdf(pos))
print(pos.shape)
plt.show()