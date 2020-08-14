import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from scipy.stats import norm as norm_scipy
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
from sklearn.decomposition import PCA

from utils import *


############################################################################################

set_plotting()

fig, ax = plt.subplots(2, 1)
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

m = 4 # n particles
w_prev = np.array([3., 1.5, 2., 3]) 
w_prev = w_prev / np.sum(w_prev)
x_prev = np.array([3.25, 4.57, 5.75, 6.5]) # also exactly the same . I think this is just bc of model mismatch
lik_center = 5. # original 2.
#----------------------------------------------------------------------------------


# weights already normalized  A GOOD ONE
# m = 4 # n particles
# w_prev = np.array([3., 1., 2., 1.]) 
# w_prev = w_prev / np.sum(w_prev)
# x_prev = np.array([3.5, 4.5, 5., 5.5]) # using 3. and 5. , with lik. in the middle, gives same solution for all
# lik_center = 4. # original 2.
#----------------------------------------------------------------------------------

# weights already normalized  PERCHE ? 
# m = 2 # n particles
# w_prev = np.array([0.5, 0.5]) 
# x_prev = np.array([3., 5]) # using 3. and 5. , with lik. in the middle, gives same solution for all
# lik_center = 4. # original 2.

#----------------------------------------------------------------------------------
# weights already normalized 
# m = 4 # n particles
# w_prev = np.array([0.03, 0.16, 0.16, 0.65]) # original [0.03, 0.16, 0.16, 0.65]
# x_prev = np.array([3., 4., 5., 6.]) # original [3., 4., 5., 6.]
# lik_center = 2. # original 2.

#----------------------------------------------------------------------------------

# m=3 # maybe 
# w_prev = np.array([0.15,0.6,0.25]) # original [0.03, 0.16, 0.16, 0.65]
# x_prev = np.array([5.25,5.5,6.25]) # original [3., 4., 5., 6.]
# lik_center = 6 # original 2.

#----------------------------------------------------------------------------------

# m=5
# w_prev = np.array([10.,3.,3.,6.,11.]) # original [0.03, 0.16, 0.16, 0.65]
# x_prev = np.array([4.5,5.5,6.,7.5,7.]) # original [3., 4., 5., 6.]
# w_prev = w_prev / np.sum(w_prev)
# lik_center = 5.5 # original 2.           THIS WORKS!!!!!!!!!!!

#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------


# m=100
# w_prev = np.random.normal(loc=10,scale=1.5,size=m)
# idxs = np.random.choice(len(w_prev), math.floor(m/2))
# w_prev[idxs]*= 10
# w_prev = w_prev / np.sum(w_prev)
# x_prev = np.random.normal(loc=5,scale=1.,size=m)
# lik_center = 4.75 # original 2.

#----------------------------------------------------------------------------------


sigma_kernels = .2 # original 0.5
sigma_lik = .7 # original 0.8 

# Neat example with sigma_lik=0.4, lik. center = 3.
# WTH example with lik_center = 4. , x_prev = np.array([3., 5., 5., 5.])

left = -3.
right = 10.

n = 100000

x = np.concatenate((np.linspace(left, right, n),x_prev))
x = np.sort(x)
X = np.array([x.tolist(),]*m)


color=iter(cm.rainbow(np.linspace(0,1,m)))

############################################################################################

# Plotting kernels requires for loop :( 
for j in range(m):
	c=next(color)
	current = w_prev[j] * norm_scipy.pdf(x, loc=x_prev[j], scale=sigma_kernels)
	ax[0].plot(x, current, c=c, label='Kernel '+str(j))

predictive = np.dot(w_prev, norm_scipy.pdf(x, loc=x_prev.reshape(-1,1), scale=sigma_kernels))

#plot likelihood 
ax[0].plot(x, norm_scipy.pdf(x, loc=lik_center, scale=sigma_lik), 'g--')

# True posterior
true_post_unnormalized = norm_scipy.pdf(x, loc=lik_center, scale=sigma_lik) * predictive
# Normalize posterior
true_post = true_post_unnormalized / (simps(true_post_unnormalized, dx=x[1]-x[0]))

# BPF proposal 
bpf_proposal = predictive

# Predictive likelihood approx
pred_lik = norm_scipy.pdf(x_prev, loc=lik_center, scale=sigma_lik)

apf_lambda = pred_lik * w_prev
apf_lambda = apf_lambda / np.sum(apf_lambda)
apf_proposal = np.dot(apf_lambda, norm_scipy.pdf(X, loc=x_prev.reshape(-1,1), scale=sigma_kernels))

# IAPF 
sum_denominator = np.sum( [ norm_scipy.pdf(x_prev, loc=x_prev[k], scale=sigma_kernels) for k in range(m)], axis=1)
sum_numerator = np.sum( [  w_prev[k] *  norm_scipy.pdf(x_prev, loc=x_prev[k], scale=sigma_kernels) for k in range(m)], axis=0) # no idea why here works with 0 and not 1

assert sum_numerator.shape == sum_denominator.shape == (m,)


iapf_lambda = ( pred_lik * sum_numerator) / sum_denominator
iapf_lambda = iapf_lambda / np.sum(iapf_lambda)
iapf_proposal = np.dot(iapf_lambda, norm_scipy.pdf(X, loc=x_prev.reshape(-1,1), scale=sigma_kernels))


eps = 1e-8

# new apf 
F1 = norm_scipy.pdf(x_prev, loc=x_prev.reshape(-1,1), scale=sigma_kernels) 

if not check_symmetric(F1):
	print('not symm')
	sys.exit()

F2 = np.vstack(( [np.array([ pred_lik[l] *  norm_scipy.pdf(x_prev[j], loc=x_prev[l], scale=sigma_kernels) for l in range(m) ])] for j in range(m) ))
F2 = pred_lik.reshape(-1,1)  * norm_scipy.pdf(x_prev, loc=x_prev.reshape(-1,1), scale=sigma_kernels)



b = np.dot(F2,w_prev) 
A = F1


A = np.hstack((A, -np.eye(b.shape[0])))
c = np.concatenate(( np.zeros(b.shape[0]), np.ones(b.shape[0])  ))
results = linprog(c=c, A_eq=A, b_eq=b, bounds=[(0,None)]*b.shape[0]*2, method='revised simplex',options={'presolve':True,'disp':True,'sparse':True}) # ,options={'presolve':False} can be interior-point or revised simplex
result = "\n Success! \n" if results['status'] == 0 else "\n Something went wrong :( \n " 
print(result)
result_vec = results['x']
psi = result_vec[:b.shape[0]]
error = np.sum(result_vec[b.shape[0]:])
print("Total error ", error)


# psi = nnls(A,b)[0]


psi = psi / np.sum(psi)
psi = normalize(psi)


new_proposal = np.dot( psi,  norm_scipy.pdf(X, loc=x_prev.reshape(-1,1), scale=sigma_kernels)  )


############################################################################################

assert np.isclose(np.sum(apf_lambda),1.)
assert np.isclose(np.sum(psi),1.)

assert np.isclose(simps(true_post, dx=x[1]-x[0]),1.)
print(simps(bpf_proposal, dx=x[1]-x[0]))
assert np.isclose(simps(bpf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(apf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(iapf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(new_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)

ax[1].plot(x,bpf_proposal, '--', c='b', label='BPF proposal')
ax[1].plot(x,apf_proposal, '--', c='y',label='APF proposal')
ax[1].plot(x,iapf_proposal, '--', c='g',label='IAPF proposal')
ax[1].plot(x,new_proposal, '--', c='m',label='New proposal')
ax[1].plot(x,true_post, '-.', c='k',label='True')

# plot posterior at selected eval points 
indices = np.array([ np.where(x == x_prev[i]) for i in range(m)]).flatten().tolist()

from sklearn.cluster import MeanShift

clustering = MeanShift().fit(true_post[indices].reshape(-1,1))

# ax[0].scatter(x_prev,true_post[indices],c=clustering.labels_)





# for i in range(5):
# 	ax[i].plot(x,true_post, c='k',label='True')

ax[0].legend(['Kernel 1', 'Kernel 2', 'Kernel 3', 'Kernel 4', 'Likelihood'])

ax[1].legend(['BPF Proposal', 'APF Proposal', 'IAPF Proposal', 'New Proposal', 'True'])


print("Chi-square for BPF: ", chi_square(true_post,bpf_proposal,x))
print("Chi-square for APF: ", chi_square(true_post,apf_proposal,x))
print("Chi-square for IAPF: ", chi_square(true_post,iapf_proposal,x))
print("Chi-square for NPF: ", chi_square(true_post,new_proposal,x))

plt.savefig("imgs/plot3.pdf", bbox_inches='tight')
