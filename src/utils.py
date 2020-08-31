import numpy as np
from scipy.integrate import simps
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from scipy.optimize import nnls,linprog,minimize
from sklearn.decomposition import PCA,TruncatedSVD
import sys
# from numpy.linalg import norm as norm_v

from random import normalvariate
from math import sqrt
from math import pi

from decimal import Decimal
# from mpmath import *
# from bigfloat import *

# mp.dps = 500

random_seed = 5

np.random.seed(random_seed)

# expp = np.vectorize(exp)
# logg = np.vectorize(log)

# robustdiv = np.vectorize(fdiv)
# bigfl = np.vectorize(BigFloat)
# nnstr = np.vectorize(nstr)


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
            'legend.loc': 'lower left'
            }
    rcParams['agg.path.chunksize'] = 10000
    rcParams.update(params)
    plt.style.use('bmh')

# works , but useless 
def cost(log_params,logA,logb):

	with precision(300):

	   # print(log_params)

	   left = np.logaddexp(  logmatmulexp(logA, log_params) , - logb).reshape(1,-1) 

	   # print(left)
	   
	   right = np.logaddexp(  logmatmulexp(logA, log_params), - logb   ) 

	   # print(right)

	   res = logmatmulexp( left, right )

	   # print(np.exp(res))

	   return res


# def cost(log_params,logA,logb):

# 	pred = logmatmulexp(logA,log_params)

# 	numerator = 2 * np.logaddexp( logb , - pred  )

# 	denominator = pred

# 	return  np.sum(numerator - denominator)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def chi_square(target,proposal,x):
  return simps( (target - proposal)**2 /(proposal), dx=x[1]-x[0])

def mse(x,y):
	return np.average((x - y)**2,axis=0)

def sparsity(x):
	return 100. - ( ( float(np.count_nonzero(x)) / float( x.size ) ) * 100 )

def robust_normalize(unnormalized):
	# print(type(fsum(unnormalized)))
	print(np.log1p(unnormalized))
	sys.exit()
	# res = unnormalized / denominator
	return res.tolist() 

def normalize(unnormalized):
	return unnormalized / np.sum(unnormalized)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

# also for logs 
def normalize_log(l):  
	return np.exp(l - logsumexp(l)).flatten()


def logmatmulexp(log_A, log_B):
    """Given matrix log_A of shape ϴ×R and matrix log_B of shape R×I, calculates                                                                                                                                                             
    (log_A.exp() @ log_B.exp()).log() in a numerically stable way.                                                                                                                                                                           
    Has O(ϴRI) time complexity and space complexity."""
    
    if len(log_B.shape) == 1:
    	log_B = log_B.reshape(-1,1)

    ϴ, R = log_A.shape
    I = log_B.shape[1]
    assert log_B.shape == (R, I)
    log_A_expanded = np.broadcast_to(np.expand_dims(log_A, 2), (ϴ, R, I))
    log_B_expanded = np.broadcast_to(np.expand_dims(log_B, 0), (ϴ, R, I))
    log_pairwise_products = log_A_expanded + log_B_expanded  # shape: (ϴ, R, I)

    if log_B.shape[1] == 1:
    	return logsumexp(log_pairwise_products, axis=1).flatten()

    return logsumexp(log_pairwise_products, axis=1)

