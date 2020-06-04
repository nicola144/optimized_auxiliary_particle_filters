import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from scipy.stats import norm
from matplotlib.pyplot import cm
from scipy.integrate import simps
from scipy.stats import multivariate_normal
import math


# Set plotting
params = {
        'axes.labelsize': 18,
        'font.size': 18,
        'legend.fontsize': 14,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'text.usetex': False,
        'figure.figsize': [15,10],
        'axes.labelpad' : 10,
        'lines.linewidth' : 10,
        'legend.loc': 'upper left'
        }
rcParams['agg.path.chunksize'] = 10000
rcParams.update(params)
plt.style.use('bmh')


fig, ax = plt.subplots(4, 1)
fig.tight_layout(pad=0.3)

# weights already normalized 
w_prev = np.array([0.03, 0.16, 0.16, 0.65]) # [0.03, 0.16, 0.16, 0.65]
x_prev = np.array([3., 4., 5., 6.]) # original [3., 4., 5., 6.]

lik_center = 2. # original 2.

sigma_kernels = 0.5 # original 0.5
sigma_lik = 0.8 # original 0.8 

# sigmoid = lambda x: (1)/(1 + np.exp(-x))

# Neat example with sigma_lik=0.4, lik. center = 3.

kernels = []

left = -10.
right = 10.

n = 100

m = len(x_prev)

x = np.linspace(left, right, n)

color=iter(cm.rainbow(np.linspace(0,1,len(w_prev))))

kernels = np.array( [norm.pdf(x, loc=x_prev[j], scale=sigma_kernels) for j in range(m)] )


# pred = np.dot(kernels, w_prev)