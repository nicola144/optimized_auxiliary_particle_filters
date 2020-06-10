import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from scipy.stats import norm
from matplotlib.pyplot import cm
from scipy import linalg
from scipy.integrate import simps
from numpy.linalg import matrix_rank
from scipy.linalg import null_space
from scipy.optimize import linprog,nnls
import math
import sys
from tqdm import tqdm
import time
from scipy.sparse import csr_matrix,csc_matrix
from sklearn import random_projection

from utils import *


############################################################################################

set_plotting()

fig, ax = plt.subplots(5, 1)
fig.tight_layout(pad=0.3)



# m = 4 # n particles
# w_prev = np.array([3., 1., 2., 3.]) 
# w_prev = w_prev / np.sum(w_prev)
# # x_prev = np.array([2.5, 3.25, 3.75, 3.]) # this with sigma lik = 1.5 means use BPF !! 
# x_prev = np.array([1.5, 3., 4.5, 5.5]) # INTERESTING . Try it with both high lik and low lik. High lik & kernels no overlap : USE APF

# lik_center = 4. # original 2.
#----------------------------------------------------------------------------------

# m = 4 # n particles
# w_prev = np.array([3., 1., 2., 1.]) 
# w_prev = w_prev / np.sum(w_prev)
# x_prev = np.array([3.5, 3.5, 3.5, 3.5]) # also exactly the same . I think this is just bc of model mismatch
# lik_center = 5. # original 2.
#----------------------------------------------------------------------------------

# weights already normalized 
# m = 4 # n particles
# w_prev = np.array([3., 1., 2., 1.]) 
# w_prev = w_prev / np.sum(w_prev)
# x_prev = np.array([3.5, 4.5, 5., 5.5]) # using 3. and 5. , with lik. in the middle, gives same solution for all
# lik_center = 4. # original 2.
#----------------------------------------------------------------------------------

# weights already normalized 
# m = 2 # n particles
# w_prev = np.array([0.5, 0.5]) 
# x_prev = np.array([3.5, 4.5]) # using 3. and 5. , with lik. in the middle, gives same solution for all
# lik_center = 3. # original 2.

#----------------------------------------------------------------------------------

# weights already normalized 
m = 4 # n particles
w_prev = np.array([0.03, 0.16, 0.16, 0.65]) # original [0.03, 0.16, 0.16, 0.65]
x_prev = np.array([3., 4., 5., 6.]) # original [3., 4., 5., 6.]
lik_center = 4. # original 2.

#----------------------------------------------------------------------------------

# m=3
# w_prev = np.array([0.1,0.6,0.3]) # original [0.03, 0.16, 0.16, 0.65]
# x_prev = np.array([4.,5.,7.]) # original [3., 4., 5., 6.]
# lik_center = 6.5 # original 2.

#----------------------------------------------------------------------------------

# m=5
# w_prev = np.array([10.,3.,3.,6.,11.]) # original [0.03, 0.16, 0.16, 0.65]
# x_prev = np.array([4.5,5.5,6.,7.5,7.]) # original [3., 4., 5., 6.]
# w_prev = w_prev / np.sum(w_prev)
# lik_center = 5. # original 2.

#----------------------------------------------------------------------------------

# m=1000
# w_prev = np.random.normal(loc=10,scale=1.5,size=m)
# idxs = np.random.choice(len(w_prev), math.floor(m/10))
# w_prev[idxs]*= 10
# w_prev = w_prev / np.sum(w_prev)
# x_prev = np.random.normal(loc=5,scale=1.,size=m)
# lik_center = 5. # original 2.

#----------------------------------------------------------------------------------


sigma_kernels = 0.5 # original 0.5
sigma_lik = 0.8 # original 0.8 

# Neat example with sigma_lik=0.4, lik. center = 3.
# WTH example with lik_center = 4. , x_prev = np.array([3., 5., 5., 5.])

left = -10.
right = 10.

n = 1000

x = np.linspace(left, right, n)
X = np.array([np.linspace(left, right, n).tolist(),]*m)

color=iter(cm.rainbow(np.linspace(0,1,m)))

############################################################################################

# Plotting kernels requires for loop :( 
for j in range(m):
	c=next(color)
	current = w_prev[j] * norm.pdf(x, loc=x_prev[j], scale=sigma_kernels)
	ax[0].plot(x, current, c=c, label='Kernel '+str(j))

predictive = np.dot(w_prev, norm.pdf(x, loc=x_prev.reshape(-1,1), scale=sigma_kernels))

#plot likelihood 
ax[0].plot(x, norm.pdf(x, loc=lik_center, scale=sigma_lik), 'g--')

# True posterior
true_post_unnormalized = norm.pdf(x, loc=lik_center, scale=sigma_lik) * predictive
# Normalize posterior
true_post = true_post_unnormalized / (simps(true_post_unnormalized, dx=x[1]-x[0]))

# BPF proposal 
bpf_proposal = predictive

# Predictive likelihood approx
pred_lik = norm.pdf(x_prev, loc=lik_center, scale=sigma_lik)

apf_lambda = pred_lik * w_prev
apf_lambda = apf_lambda / np.sum(apf_lambda)
apf_proposal = np.dot(apf_lambda, norm.pdf(X, loc=x_prev.reshape(-1,1), scale=sigma_kernels))

# IAPF 
sum_denominator = np.zeros((m,))
for i in range(m):
  for j in range(m):
    sum_denominator[i]+= pred_lik[j] * norm.pdf(x_prev[i], x_prev[j], scale=sigma_kernels)

# sum_denominator = np.sum( [norm.pdf(x_prev, loc=x_prev[k], scale=sigma_kernels) for k in range(m)], axis=1)
# cambiato
sum_numerator = np.sum( [   w_prev[k] * pred_lik[k] * norm.pdf(x_prev, loc=x_prev[k], scale=sigma_kernels) for k in range(m)], axis=0) # no idea why here works with 0 and not 1

iapf_lambda = (pred_lik * sum_numerator) / sum_denominator
iapf_lambda = iapf_lambda / np.sum(iapf_lambda)
iapf_proposal = np.dot(iapf_lambda, norm.pdf(X, loc=x_prev.reshape(-1,1), scale=sigma_kernels))


eps = 1e-8

# new apf 
F1 = norm.pdf(x_prev, loc=x_prev.reshape(-1,1), scale=sigma_kernels) 
print(F1)
print(norm.pdf(x_prev[0], loc=x_prev[0], scale=sigma_kernels))
print(norm.pdf(x_prev[0], loc=x_prev[1], scale=sigma_kernels))
print(norm.pdf(x_prev[0], loc=x_prev[2], scale=sigma_kernels))
print(norm.pdf(x_prev[0], loc=x_prev[3], scale=sigma_kernels))

sys.exit()
# F2 = np.vstack(( [np.array([ pred_lik[l] *  norm.pdf(x_prev[j], loc=x_prev[l], scale=sigma_kernels) for l in range(m) ])] for j in range(m) ))
F2 = pred_lik * norm.pdf(x_prev, loc=x_prev.reshape(-1,1), scale=sigma_kernels) 

# F1[np.abs(F2) < eps] = 0.
# F1[np.abs(F2) < eps] = 0.
# perc1 = (float(np.count_nonzero(F1)) / float(F1.shape[0] * F1.shape[1] ))*100.
# print('\n')
# print("Percentage of zeros for F1: ", 100. - perc1, "\n")
# perc2 = (float(np.count_nonzero(F1)) / float(F1.shape[0] * F1.shape[1] ))*100.
# print('\n')
# print("Percentage of zeros for F2: ", 100. - perc2, "\n")

# const = .5
# F1+= const * np.eye(F1.shape[0], F1.shape[1])
# F2+= const * np.eye(F2.shape[0], F2.shape[1])

b = np.dot(F2.T,w_prev) 
A = F1.T

# A[np.abs(A) < eps] = 0.
# A = np.dot(A.T,A) + np.eye(A.shape[0])
# b = np.dot(A.T,b)
# print('start svd')
# U,s,V = linalg.svd(A)

# print("Percentage of zeros for A: ", 100. -  (float(np.count_nonzero(A)) / float(A.shape[0] * A.shape[1] ))*100.  , "\n")
A[np.abs(A) < eps] = 0.

A = np.hstack((A, -np.eye(b.shape[0])))
c = np.concatenate(( np.zeros(b.shape[0]), np.ones(b.shape[0])  ))
results = linprog(c=c, A_eq=A, b_eq=b, bounds=[(0,None)]*b.shape[0]*2, method='revised simplex',options={'presolve':True,'disp':True,'sparse':True}) # ,options={'presolve':False} can be interior-point or revised simplex
result = "\n Success! \n" if results['status'] == 0 else "\n Something went wrong :( \n " 
print(result)
result_vec = results['x']
psi = result_vec[:b.shape[0]]
error = np.sum(result_vec[b.shape[0]:])
print("Total error ", error)

# print(A.shape)
# transformer = random_projection.GaussianRandomProjection()
# A = transformer.fit_transform(A)
# print(A.shape)

# psi = nnls(A,b)[0]


# U,s,V = linalg.svd(F)
# psi =  V[-1].clip(min=0) + w_prev
# psi = (pred_lik) * psi
psi[np.abs(psi) < eps] = 0.
nnz = np.count_nonzero(psi)
perc = (float(nnz) / float(psi.shape[0]))*100.
print('\n')
print("Percentage of zeros in lambdas: ", 100. - perc, "\n")

psi = psi / np.sum(psi)

# new_proposal = np.sum( psi.reshape(-1,1) * norm.pdf(X, loc=x_prev.reshape(-1,1), scale=sigma_kernels), axis=0)
new_proposal = np.dot( psi,  norm.pdf(X, loc=x_prev.reshape(-1,1), scale=sigma_kernels)  )

############################################################################################

assert np.isclose(np.sum(apf_lambda),1.)
assert np.isclose(np.sum(psi),1.)

assert np.isclose(simps(true_post, dx=x[1]-x[0]),1.)
assert np.isclose(simps(bpf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(apf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(iapf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(new_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)

ax[1].plot(x,bpf_proposal, c='b', label='BPF proposal')
ax[2].plot(x,apf_proposal, c='y',label='APF proposal')
ax[3].plot(x,iapf_proposal, c='g',label='IAPF proposal')
ax[4].plot(x,new_proposal, c='m',label='NEW proposal')

for i in range(5):
	ax[i].plot(x,true_post, c='k',label='True')

ax[1].legend(['BPF Proposal', 'True'])
ax[2].legend(['APF Proposal', 'True'])
ax[3].legend(['IAPF Proposal', 'True'])
ax[4].legend(['NEW Proposal', 'True'])


print("Chi-square for BPF: ", chi_square(true_post,bpf_proposal,x))
print("Chi-square for APF: ", chi_square(true_post,apf_proposal,x))
print("Chi-square for IAPF: ", chi_square(true_post,iapf_proposal,x))
print("Chi-square for NPF: ", chi_square(true_post,new_proposal,x))

plt.show()
# plt.savefig("test2.png", bbox_inches='tight')
