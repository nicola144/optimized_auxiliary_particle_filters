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

# from mpmath import *
# from bigfloat import *

# mp.dps = 500

random_seed = 1

np.random.seed(random_seed)

# expp = np.vectorize(exp)
# logg = np.vectorize(log)

# robustdiv = np.vectorize(fdiv)
# bigfl = np.vectorize(BigFloat)
# nnstr = np.vectorize(nstr)


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


def log_like_iid_nbinom_log_params(log_lambda, n):
    """Log likelihood for i.i.d. Gasussian measurements with
    input being logarithm of parameters.

    Parameters
    ----------
    log_params : array
        Logarithm of the parameters alpha and b.
    n : array
        Array of counts.

    Returns
    -------
    output : float
        Log-likelihood.
    """
    log_alpha, log_b = log_params

    alpha = np.exp(log_alpha)
    b = np.exp(log_b)

    return np.sum(st.nbinom.logpmf(n, alpha, 1/(1+b)))


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
	        'legend.loc': 'upper left'
	        }
	rcParams['agg.path.chunksize'] = 10000
	rcParams.update(params)
	plt.style.use('bmh')


def flattenn(l):
	return [item for sublist in l for item in sublist]


def randomUnitVector(n):
    # unnormalized = [normalvariate(0, 1) for _ in range(n)]
    unnormalized = flattenn([np.random.lognormal(0,1,size=1).tolist() for _ in range(n)])
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    n, m = A.shape
    x = randomUnitVector(min(n,m))
    lastV = None
    currentV = x

    # print(np.log(A))
    # sys.exit()
    if not np.all(A >= 0.):
        print('neg A')
        sys.exit()

    if n > m:
        B = np.dot(A.T, A)
    else:
        print('here')
        B = logmatmulexp(np.log(A.T), np.log(A))
        print('done ')

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = logmatmulexp(B, np.log(lastV))
        currentV = np.exp(currentV) / norm_v(np.exp(currentV))

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV


def svd(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    A = np.array(A, dtype=float)
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            if not np.all(u >= 0.) or not np.all(v >= 0.):
                print('neg')
                sys.exit()
            # matrixFor1D -= singularValue * np.outer(u, v)
            logu = np.log(u.reshape(-1,1))
            logv = np.log(v[np.newaxis,...])
            decrement = np.log(singularValue) + logmatmulexp(logu,logv)
            # out of logspace just for this
            matrixFor1D-=np.exp(decrement)

        if not np.all(matrixFor1D >= 0.):
        	print('lready')
        	sys.exit()

        if n > m:
            v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = norm_v(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.exp(logmatmulexp(np.log(A.T), np.log(u)))
            sigma = norm_v(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs


# def randomUnitVector(n):
#     # unnormalized = [normalvariate(0, 1) for _ in range(n)]
#     unnormalized = flattenn([np.random.lognormal(0,1,size=1).tolist() for _ in range(n)])

#     theNorm = sum(x * x for x in unnormalized)
#     # dont really need the sqrt 
#     return [x / theNorm for x in unnormalized]


# def svd_1d(A, epsilon=1e-10):
#     ''' The one-dimensional SVD '''
#     # A = np.exp(A)
#     n, m = A.shape
#     x = np.log(randomUnitVector(min(n,m)))
#     lastV = None
#     currentV = x

#     if n > m:
#         print('no')
#         B = np.dot(A.T, A)
#     else:
#         # B = np.dot(A, A.T)
#         B = logmatmulexp(A, A.T)

#     iterations = 0
#     while True:
#         iterations += 1
#         lastV = currentV
#         # currentV = np.dot(B, lastV)
#         # currentV = currentV / norm_v(currentV)
#         currentV = logmatmulexp(B, lastV)
#         currentV = np.logaddexp( currentV, -np.log(norm_v( np.exp(currentV) ))  )

#         # if abs(np.dot(currentV, lastV)) > 1 - epsilon:
#         if abs(logmatmulexp(currentV.reshape(1,-1), lastV.reshape(-1,1))) > 1 - epsilon:
#             print("converged in {} iterations!".format(iterations))
#             return currentV


def addlog(logA,logB):
	assert logA.shape == logB.shape

	# logA = np.log(A)
	# logB = np.log(B)

	n = logA.shape[0]
	m = logA.shape[1]

	res = np.empty((n,m))
	for i in range(n):
		for j in range(m):
			a = np.array([logA[i][j], logB[i][j]]).reshape(1,-1)
			b = np.log(np.array([1,1])).reshape(-1,1)
			res[i][j] = logmatmulexp(a,b)

	return res

# # def nnnorm_v(x):
# # 	return x.dot(x)

# def svd(A, k=None, epsilon=1e-10):
#     '''
#         Compute the singular value decomposition of a matrix A
#         using the power method. A is the input matrix, and k
#         is the number of singular values you wish to compute.
#         If k is None, this computes the full-rank decomposition.
#     '''
#     A = np.array(A, dtype=float)
#     n, m = A.shape
#     svdSoFar = []
#     if k is None:
#         k = min(n, m)

#     for i in range(k):
#         matrixFor1D = np.log(A.copy())
#         # matrixFor1D = A.copy()
#         # done_first = False

#         for singularValue, u, v in svdSoFar[:i]:

#             # matrixFor1D -= singularValue * np.matmul(u.reshape(-1,1),v[np.newaxis,...])

#             logu = np.log(u.reshape(-1,1))
#             logv = np.log(v[np.newaxis,...])
#             decrement = np.logaddexp(np.log(singularValue),logmatmulexp(logu,logv))
#             matrixFor1D = addlog(matrixFor1D, -decrement)

#             # if not done_first:
#             # 	prev = matrixFor1D
#             # 	done_first = True
#             # else:
#             # 	prev = np.log(matrixFor1D)
#             # matrixFor1D = np.exp( addlog( prev , - ( decrement ) ) )



#         if n > m:
#             v = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
#             u_unnormalized = np.dot(A, v)
#             sigma = norm_v(u_unnormalized)  # next singular value
#             u = u_unnormalized / sigma
#         else:
#             u = svd_1d(matrixFor1D, epsilon=epsilon)  # next singular vector
#             # v_unnormalized = np.dot(A.T, u)
#             v_unnormalized = logmatmulexp(np.log(A.T), u )

#             sigma = np.log(norm_v( np.exp(v_unnormalized) ))  # next singular 

#             # v = v_unnormalized / sigma
#             v = np.logaddexp(v_unnormalized, -sigma)

#         svdSoFar.append((sigma, u, v))

#     singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
#     return np.exp(singularValues), us.T, vs
