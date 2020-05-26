import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm
from scipy.stats import norm
from matplotlib.pyplot import cm
from scipy.integrate import simps
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
sigma_lik = 0.8 # original 0.8 . 

sigmoid = lambda x: (1)/(1 + np.exp(-x))

# Neat example with sigma_lik=0.4, lik. center = 3.

kernels = []

left = -10.
right = 10.

n = 100

x = np.linspace(left, right, n)

color=iter(cm.rainbow(np.linspace(0,1,len(w_prev))))

pred = np.zeros((n,))

for j in range(len(w_prev)):
	c=next(color)
	current = w_prev[j] * norm.pdf(x, loc=x_prev[j], scale=sigma_kernels)
	pred+= current
	ax[0].plot(x, current, c=c, label='Kernel '+str(j))

#plot likelihood 
ax[0].plot(x, norm.pdf(x, loc=lik_center, scale=sigma_lik), 'g--')

# True posterior
true_post_unnormalized = norm.pdf(x, loc=lik_center, scale=sigma_lik) * pred
# Normalize posterior
true_post = true_post_unnormalized / (simps(true_post_unnormalized, dx=x[1]-x[0]))

# BPF proposal 
bpf_proposal = pred

# Predictive likelihood approx
pred_lik = []

for j in range(len(x_prev)):
	pred_lik.append(norm.pdf(x_prev[j], loc=lik_center, scale=sigma_lik))	
	#other approx:
	# K = 50.
	# samples = norm.rvs(loc=x_prev[j], scale=sigma_kernels,size=int(K))
	# cumsum = 0.
	# for k in range(int(K)):
	# 	cumsum+= norm.pdf(samples[k], loc=lik_center, scale=sigma_lik)
	# pred_lik.append((1./K) * cumsum)


pred_lik = np.array(pred_lik)
apf_weights = pred_lik * w_prev 
apf_weights = apf_weights/(np.sum(apf_weights))

# APF
apf_proposal = np.zeros((n,))

for j in range(len(x_prev)):
	apf_proposal+= apf_weights[j] * norm.pdf(x, loc=x_prev[j], scale=sigma_kernels)

# IAPF 
# sum_denominator = 0.
# sum_numerator = 0.

# for i in range(len(x_prev)):
# 	sum_denominator+=  norm.pdf(x_prev[i], loc=lik_center, scale=sigma_kernels)
# 	sum_numerator+= w_prev[i] * norm.pdf(x_prev[i], loc=lik_center, scale=sigma_kernels)

sum_denominator = np.zeros((pred_lik.shape[0],))
sum_numerator = np.zeros((pred_lik.shape[0],))

for j in range(len(x_prev)):
	for k in range(len(x_prev)):
		sum_denominator[j]+= norm.pdf(x_prev[j], loc=x_prev[k], scale=sigma_kernels)
		sum_numerator[j]+= w_prev[k] * norm.pdf(x_prev[j], loc=x_prev[k], scale=sigma_kernels)

# iapf_weights = ( pred_lik * sum_numerator ) / (sum_denominator * (x_prev[j] - lik_center)**2 )
iapf_weights = (pred_lik * sum_numerator) / sum_denominator
iapf_weights = iapf_weights / np.sum(iapf_weights)

iapf_proposal = np.zeros((n,))

for j in range(len(x_prev)):
	# d = np.absolute(x_prev[j] - lik_center)
	# phi = sigmoid(d)
	# iapf_proposal+= iapf_weights[j] * ( (1-phi) * norm.pdf(x, loc=x_prev[j], scale=sigma_kernels) + phi * norm.pdf(x, loc=lik_center,scale=sigma_lik) )
	iapf_proposal+= iapf_weights[j] * norm.pdf(x, loc=x_prev[j], scale=sigma_kernels)

assert np.isclose(np.sum(apf_weights),1.)
assert np.isclose(np.sum(iapf_weights),1.)

ax[1].plot(x,bpf_proposal, c='b', label='BPF proposal')
ax[2].plot(x,apf_proposal, c='y',label='APF proposal')
ax[3].plot(x,iapf_proposal, c='g',label='IAPF proposal')

for i in range(4):
	ax[i].plot(x,true_post, c='k',label='True')

ax[0].legend(['Kernel 0', 'Kernel 1', 'Kernel 2', 'Kernel 3', 'Likelihood', 'True'])
ax[1].legend(['BPF Proposal', 'True'])
ax[2].legend(['APF Proposal', 'True'])
ax[3].legend(['IAPF Proposal', 'True'])

assert np.isclose(simps(true_post, dx=x[1]-x[0]),1.)
assert np.isclose(simps(bpf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
assert np.isclose(simps(apf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)
# assert np.isclose(simps(iapf_proposal, dx=x[1]-x[0]),1.,rtol=1e-2,atol=1e-2)

plt.show()
# plt.savefig("iapf.png", bbox_inches='tight',format='png')