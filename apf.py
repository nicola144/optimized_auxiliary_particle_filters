import cupy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from scipy.stats import norm
from matplotlib.pyplot import cm
from scipy import linalg
from scipy.integrate import simps
from numpy.linalg import matrix_rank
from scipy.linalg import null_space
from scipy.optimize import linprog
import math
import sys
from tqdm import tqdm
import time
from scipy.sparse import csr_matrix,csc_matrix


############################################################################################
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

fig, ax = plt.subplots(5, 1)
fig.tight_layout(pad=0.3)

#----------------------------------------------------------------------------------

# weights already normalized 
# m = 4 # n particles
# w_prev = np.array([0.03, 0.16, 0.16, 0.65]) # original [0.03, 0.16, 0.16, 0.65]
# x_prev = np.array([3., 4., 5., 6.]) # original [3., 4., 5., 6.]
# lik_center = 6. # original 2.

#----------------------------------------------------------------------------------

# m=3
# w_prev = np.array([0.1,0.6,0.3]) # original [0.03, 0.16, 0.16, 0.65]
# x_prev = np.array([4.,5.,7.]) # original [3., 4., 5., 6.]
# lik_center = 6.5 # original 2.

#----------------------------------------------------------------------------------

# m=10
# w_prev = np.array([0.3,0,0.05]) # original [0.03, 0.16, 0.16, 0.65]
# x_prev = np.array([4.,5.,7.]) # original [3., 4., 5., 6.]
# lik_center = 6.5 # original 2.

#----------------------------------------------------------------------------------

m=10000
w_prev = np.random.normal(loc=10,scale=1.5,size=m)
idxs = np.random.choice(len(w_prev), math.floor(m/500))
w_prev[idxs]*= 100000
w_prev = w_prev / np.sum(w_prev)
x_prev = np.random.normal(loc=5,scale=1.,size=m)
lik_center = 5. # original 2.

#----------------------------------------------------------------------------------


sigma_kernels = 0.5 # original 0.5
sigma_lik = 0.8 # original 0.8 

# Neat example with sigma_lik=0.4, lik. center = 3.
# WTH example with lik_center = 4. , x_prev = np.array([3., 5., 5., 5.])

left = -10.
right = 10.

n = 100

x = np.linspace(left, right, n)
X = np.array([np.linspace(left, right, n).tolist(),]*m)

color=iter(cm.rainbow(np.linspace(0,1,m)))

############################################################################################

# Plotting kernels requires for loop :( 
# for j in range(m):
# 	c=next(color)
# 	current = w_prev[j] * norm.pdf(x, loc=x_prev[j], scale=sigma_kernels)
# 	ax[0].plot(x, current, c=c, label='Kernel '+str(j))

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
sum_denominator = np.sum( [norm.pdf(x_prev, loc=x_prev[k], scale=sigma_kernels) for k in range(m)], axis=1 )
sum_numerator = np.sum( [ w_prev[k] * norm.pdf(x_prev, loc=x_prev[k], scale=sigma_kernels) for k in range(m)], axis=0) # no idea why here works with 0 and not 1

iapf_lambda = (pred_lik * sum_numerator) / sum_denominator
iapf_lambda = iapf_lambda / np.sum(iapf_lambda)
iapf_proposal = np.dot(iapf_lambda, norm.pdf(X, loc=x_prev.reshape(-1,1), scale=sigma_kernels))


eps = 1e-8

# new apf 
F1 = norm.pdf(x_prev, loc=x_prev.reshape(-1,1), scale=sigma_kernels) 
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

# A_sparse = csc_matrix(A,shape=A.shape)
# assert A_sparse.shape == A.shape

results = linprog(c=c, A_eq=A, b_eq=b, bounds=[(0,None)]*b.shape[0]*2, method='revised simplex',options={'presolve':True,'disp':True,'sparse':True}) # ,options={'presolve':False} can be interior-point or revised simplex
result = "\n Success! \n" if results['status'] == 0 else "\n Something went wrong :( \n " 
print(result)
result_vec = results['x']
psi = result_vec[:b.shape[0]]
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

ax[1].plot(x,bpf_proposal, c='b', label='BPF proposal')
ax[2].plot(x,apf_proposal, c='y',label='APF proposal')
ax[3].plot(x,iapf_proposal, c='g',label='IAPF proposal')
ax[4].plot(x,new_proposal, c='m',label='NEW proposal')

for i in range(5):
	ax[i].plot(x,true_post, c='k',label='True')

# ax[0].legend(['Kernel 0', 'Kernel 1', 'Kernel 2', 'Kernel 3', 'Likelihood', 'True'])
ax[1].legend(['BPF Proposal', 'True'])
ax[2].legend(['APF Proposal', 'True'])
ax[3].legend(['IAPF Proposal', 'True'])
ax[4].legend(['NEW Proposal', 'True'])

assert np.isclose(simps(true_post, dx=x[1]-x[0]),1.)
assert np.isclose(simps(bpf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(apf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(iapf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(new_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)

# plt.show()
plt.savefig("test2.png", bbox_inches='tight')