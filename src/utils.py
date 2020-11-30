import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.integrate import simps
from scipy.special import logsumexp
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time
import random
from tqdm import tqdm

# random_seed = 5

def compute_cluster_centers(all_points):
	all_points = StandardScaler().fit_transform(all_points)
	db = DBSCAN(eps=0.3, min_samples=10).fit(all_points)
	labels = db.labels_
	print(labels)
	sys.exit()
	labels_unique = np.unique(db.labels_)

	print(labels_unique.shape)
	sys.exit()

	centroids = []
	for i in range(len(labels_unique)):
		centroids.append(np.mean(all_points[labels_unique==i,:], axis=0))

	centroids = np.asarray(centroids)
	print(centroids.shape)
	sys.exit()

# Implements the IHS update. "right" means the term on the right of the difference
# in the update rule (same for "left")
def ihs_step(lamb, A, b, S, old_lambda, K):
	right = (A.T.dot(b - A.dot(old_lambda))).T.dot(lamb)
	norm_term = S.dot(A.dot(lamb - old_lambda))
	left = (1. / (2. * K)) * norm_term.T.dot(norm_term)
	res = left - right
	return res

def randomized_nnls(A, b, n_particle):
	"""
	Aim is to solve NNLS problem, using Iterative Hessian Sketching:
	find NONNEGATIVE lambda = arg min  || A . lambda - b ||_{2}^{2}

	Also, there should be at least one non zero lambda.
	"""
	K = int(n_particle / 2)
	# Initialized at 0 for iteration 0 as in Pilanci & Wainwright 2016
	lambdas = np.zeros(b.shape)

	# Result lambdas should NOT be ALL zeros.
	# cons = ({'type': 'ineq', 'fun': lambda x: x.dot(x) - 0.01})

	for i in range(5): # 5 iterations
		init_lambdas = np.random.multivariate_normal(mean=np.zeros(b.shape), cov=np.eye(b.shape[0]))

		# Sketching matrix. Using Gaussian Sketch.
		S = np.random.normal(0, 1, (K, b.shape[0]))
		S /= np.sqrt(K)

		# Minimize the IHS objective, subject to a positive result, with the added constraint as above
		res = minimize(ihs_step, init_lambdas, (A, b, S, lambdas, K),  bounds=[(0., None)] * b.shape[0])
		lambdas = res['x']

	return lambdas


def scale_reduced_system(smaller_A, smaller_b):
	smallest_exp_A = np.min(smaller_A)
	smallest_exp_b = np.min(smaller_b)
	smallest = np.min([smallest_exp_A, smallest_exp_b])
	smallest = np.format_float_scientific(smallest)
	min_exp = int(re.findall(r'\d+', smallest)[-1])
	scaled_smaller_A = smaller_A * (10 ** min_exp)
	scaled_smaller_b = smaller_b * (10 ** min_exp)

	return scaled_smaller_A, scaled_smaller_b



def safe_invert(matrix):
	return np.linalg.lstsq(matrix, np.eye(matrix.shape[0]))[0]

def reduce_system(n_particle, A, b):

	# K = int(n_particle / 50)
	K = 5
	indices_tokeep = b.argsort()[-K:][::-1]
	# indices_tokeep = np.round(np.linspace(0, b.shape[0] - 1, K)).astype(int)

	smaller_b = b[indices_tokeep]
	temp = A[:, indices_tokeep]
	smaller_A = temp[indices_tokeep, :]

	return smaller_A, smaller_b, indices_tokeep


def sanity_checks(unnormalized):
	if np.all(unnormalized == 0.):
		print('ALL zeros ... \n ')
		print(unnormalized)
		sys.exit()

	if np.isnan(np.log(unnormalized)).any():
		print(unnormalized)
		print('some log  nan')
		sys.exit()


def set_plotting():
	# Set plotting
	params = {
		'axes.labelsize': 25,
		'font.size': 20,
		'legend.fontsize': 24,
		'xtick.labelsize': 25,
		'ytick.labelsize': 25,
		'text.usetex': False,
		'figure.figsize': [20, 12],
		'axes.labelpad': 10,
		'lines.linewidth': 10,
		'legend.loc': 'upper left'
	}
	rcParams['agg.path.chunksize'] = 10000
	rcParams.update(params)
	plt.style.use('bmh')


def is_pos_def(x):
	return np.all(np.linalg.eigvals(x) > 0)


def chi_square(target, proposal, x):
	return simps((target - proposal) ** 2 / (proposal), dx=x[1] - x[0])


# def mse(x, y):
# 	return np.average((x - y) ** 2, axis=0)

# Normalized ?
def mse(x, y):
	return np.average((x - y) ** 2, axis=0) / np.average(np.sum(y**2,axis=-1))


def sparsity(x):
	return 100. - ((float(np.count_nonzero(x)) / float(x.size)) * 100)


def normalize(unnormalized):
	return unnormalized / np.sum(unnormalized)


def check_symmetric(a, rtol=1e-05, atol=1e-08):
	return np.allclose(a, a.T, rtol=rtol, atol=atol)


def normalize_log(l):
	return np.exp(l - logsumexp(l)).flatten()

def log_normalize_log(unnormalized):
	return unnormalized - logsumexp(unnormalized)

def get_ess(logw_norm):
	return np.exp(-logsumexp(2*logw_norm))


def logmatmulexp(log_A, log_B):
	"""Given matrix log_A of shape ϴ×R and matrix log_B of shape R×I, calculates
	(log_A.exp() @ log_B.exp()).log() in a numerically stable way.
	Has O(ϴRI) time complexity and space complexity."""

	if len(log_B.shape) == 1:
		log_B = log_B.reshape(-1, 1)

	ϴ, R = log_A.shape
	I = log_B.shape[1]
	assert log_B.shape == (R, I)
	log_A_expanded = np.broadcast_to(np.expand_dims(log_A, 2), (ϴ, R, I))
	log_B_expanded = np.broadcast_to(np.expand_dims(log_B, 0), (ϴ, R, I))
	log_pairwise_products = log_A_expanded + log_B_expanded  # shape: (ϴ, R, I)

	if log_B.shape[1] == 1:
		return logsumexp(log_pairwise_products, axis=1).flatten()

	return logsumexp(log_pairwise_products, axis=1)

# works , but useless 
# def cost(log_params,logA,logb):

#   with precision(300):

#      # print(log_params)

#      left = np.logaddexp(  logmatmulexp(logA, log_params) , - logb).reshape(1,-1) 

#      # print(left)

#      right = np.logaddexp(  logmatmulexp(logA, log_params), - logb   ) 

#      # print(right)

#      res = logmatmulexp( left, right )

#      # print(np.exp(res))

#      return res
