import numpy as np
from scipy.special import logsumexp
from utils import *

def sublog(logA,logB):
	assert logA.shape == logB.shape


	n = logA.shape[0]
	m = logA.shape[1]

	res = np.empty((n,m))
	for i in range(n):
		for j in range(m):
			a = np.array([logA[i][j], logB[i][j]]).reshape(1,-1)
			b = np.log(np.array([1,1])).reshape(-1,1)
			res[i][j] = logmatmulexp(a,b)

	return res


def addlog(logA,logB):
	assert logA.shape == logB.shape


	n = logA.shape[0]
	m = logA.shape[1]

	res = np.empty((n,m))
	for i in range(n):
		for j in range(m):
			a = np.array([logA[i][j], logB[i][j]]).reshape(1,-1)
			b = np.log(np.array([1,1])).reshape(-1,1)
			res[i][j] = logmatmulexp(a,b)

	return res
	# return np.exp(res)

A = np.array([[1,2,3],[4,5,6]])

B = np.array([[1,1,1],[1,1,1]])


for i in range(1):
	A = np.exp(sublog( np.log(A), np.log(B) ))

# for i in range(1):
# 	A-= B

print(A)

# print(A*3)
# print( np.exp(  np.log(3) + np.log(A) ) )