import numpy as np
from scipy.integrate import simps
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm



def chi_square(target,proposal,x):
  return simps( (target - proposal)**2 /(proposal), dx=x[1]-x[0])

def mse(x,y):
	return np.average((x - y)**2,axis=0)

# numerically stable normalization. works with logs
# def normalize(unnormalized):
#     maxx = np.max(unnormalized);
#     unnormalized = unnormalized - maxx;
#     unnormalized = np.exp(unnormalized);
#     return unnormalized / np.sum(unnormalized)

# def normalize(unnormalized):
# 	return unnormalized / np.sum(unnormalized)

# def normalize_log(l):
# 	return np.exp(l - logsumexp(l))

def logmatmulexp(log_A, log_B):
    """Given matrix log_A of shape ϴ×R and matrix log_B of shape R×I, calculates                                                                                                                                                             
    (log_A.exp() @ log_B.exp()).log() in a numerically stable way.                                                                                                                                                                           
    Has O(ϴRI) time complexity and space complexity."""
    ϴ, R = log_A.shape
    I = log_B.shape[1]
    assert log_B.shape == (R, I)
    log_A_expanded = np.broadcast_to(np.expand_dims(log_A, 2), (ϴ, R, I))
    log_B_expanded = np.broadcast_to(np.expand_dims(log_B, 0), (ϴ, R, I))
    log_pairwise_products = log_A_expanded + log_B_expanded  # shape: (ϴ, R, I)                                                                                                                                                              
    return logsumexp(log_pairwise_products, axis=1)


def set_plotting():
	# Set plotting
	params = {
	        'axes.labelsize': 18,
	        'font.size': 14,
	        'legend.fontsize': 14,
	        'xtick.labelsize': 14,
	        'ytick.labelsize': 14,
	        'text.usetex': False,
	        'figure.figsize': [18,8],
	        'axes.labelpad' : 10,
	        'lines.linewidth' : 10,
	        'legend.loc': 'lower right'
	        }
	rcParams['agg.path.chunksize'] = 10000
	rcParams.update(params)
	plt.style.use('bmh')
