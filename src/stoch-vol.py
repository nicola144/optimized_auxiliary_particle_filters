

from utils import *
from particles import *

random_state = np.random.RandomState(random_seed)

set_plotting()

# for more plots
fig, ax = plt.subplots(1, 2)
ax = ax.reshape(1,2)
fig.tight_layout(pad=0.3)
plt.subplots_adjust(top=0.95)
plt.tight_layout(pad=0.3)


dim = 2
timesteps = 100
n_particle_bpf = 100
n_particle_apf = 100
n_particle_iapf = 100
n_particle_npf = 100


constant_mean = np.zeros(dim,)
initial_cov = np.eye(dim)
transition_cov = 0.1 * np.eye(dim)

# phi = np.diag( random_state.randint(low=0,high=5,size=dim ) )
phi = np.diag(np.ones(dim,))

def prior_sample(size=1):
	res = np.random.multivariate_normal(mean=constant_mean,cov=initial_cov,size=size)
	return res.squeeze()

def transition_sample(prev_state):
	return np.random.multivariate_normal(mean=constant_mean + np.dot(phi, prev_state - constant_mean), cov=transition_cov)

def observation_sample(curr_state):
	obs_cov = np.diag(np.exp(curr_state))

	assert np.all( np.linalg.eigh( obs_cov )[0] > 0 )
	return np.random.multivariate_normal(mean=np.zeros(dim,), cov=obs_cov) 


obs = []
states = []

curr_state = prior_sample()

states.append(curr_state)

obs.append( observation_sample(curr_state) )

for t in range(timesteps - 1):
	# transition state
	curr_state = transition_sample(curr_state)

	states.append(curr_state)

	#get obs
	obs.append(observation_sample(curr_state))


observations = np.array(obs)
states = np.array(states)

bpf = StochVolBPF(init_particle=prior_sample(size=n_particle_bpf),
						random_state=random_state,
						transition_cov=transition_cov,
						transition_offset=constant_mean,
						phi=phi )

apf = StochVolAPF(init_particle=prior_sample(size=n_particle_apf),
						random_state=random_state,
						transition_cov=transition_cov,
						transition_offset=constant_mean,
						phi=phi )

iapf = StochVolIAPF(init_particle=prior_sample(size=n_particle_iapf),
						random_state=random_state,
						transition_cov=transition_cov,
						transition_offset=constant_mean,
						phi=phi )

npf = StochVolNewAPF(init_particle=prior_sample(size=n_particle_npf),
						random_state=random_state,
						transition_cov=transition_cov,
						transition_offset=constant_mean,
						phi=phi )


mean_bpf, covs_bpf, liks_bpf, ess_bpf, n_unique_bpf, w_vars_bpf = bpf.filter(observations)
mean_apf, covs_apf, liks_apf, ess_apf, n_unique_apf, w_vars_apf = apf.filter(observations)
mean_iapf, covs_iapf, liks_iapf, ess_iapf, n_unique_iapf, w_vars_iapf  = iapf.filter(observations)
mean_npf, covs_npf, liks_npf, ess_npf, n_unique_npf, w_vars_npf  = npf.filter(observations)




# with open('myfile.npy', 'wb') as f:
# 	np.save(f, mean_bpf)

# with open('myfile.npy', 'rb') as f:
# 	mean_bpf = np.load(f)

# plt.plot(ess_bpf, 'b', label='bpf')
# plt.plot(ess_apf, 'y', label='apf')
# plt.plot(ess_iapf, 'c', label='iapf')
# plt.plot(ess_npf, 'm', label='npf')
# plt.xlabel('timestep')
# plt.ylabel('ess_estimate')
# plt.show()
# plt.close()

ax[0][0].plot(w_vars_bpf, 'b', label='bpf')
ax[0][0].plot(w_vars_apf, 'y', label='apf')
ax[0][0].plot(w_vars_iapf, 'c', label='iapf')
ax[0][0].plot(w_vars_npf, 'm', label='oapf')
ax[0][0].set(xlabel="timestep",ylabel="w_var")
# ax[0][0].set_xlim()
ax[0][0].set_ylim( (0., np.max( [np.max(w_vars_apf), np.max(w_vars_iapf), np.max(w_vars_npf)] ) + 1. ))


ax[0][1].plot(ess_bpf, 'b', label='bpf')
ax[0][1].plot(ess_apf, 'y', label='apf')
ax[0][1].plot(ess_iapf, 'c', label='iapf')
ax[0][1].plot(ess_npf, 'm', label='oapf')
ax[0][1].set(xlabel="timestep",ylabel="ess")
plt.legend()

# plt.savefig('5svm.pdf', bbox_inches='tight')

plt.show()

# i = 0
# plt.plot(states[:,i],'r', label='true_state')
# plt.plot(mean_bpf[:,i],'b', label='mean_bpf')
# plt.plot(mean_npf[:,i],'m', label='mean_npf')
# plt.fill_between(np.arange(len(mean_npf[:,i])), mean_npf[:,i] - np.sqrt(covs_npf[:,i,i]), mean_npf[:,i] + np.sqrt(covs_npf[:,i,i]), edgecolor=(1 , 0.2, 0.8, 0.99) , facecolor=(1, 0.2, 0.8, 0.3), label="std_npf", linewidth=1.5)
# plt.fill_between(np.arange(len(mean_bpf[:,i])), mean_bpf[:,i] - np.sqrt(covs_bpf[:,i,i]), mean_bpf[:,i] + np.sqrt(covs_bpf[:,i,i]), edgecolor=(0 , 0, 1, 0.99) , facecolor=(0, 0, 1, 0.3), label="std_bpf", linewidth=1)
# plt.plot(mean_iapf[:,i], '2--',color='c',label='mean_iapf')
# plt.fill_between(np.arange(len(mean_iapf[:,i])), mean_iapf[:,i] - np.sqrt(covs_iapf[:,i,i]), mean_iapf[:,i] + np.sqrt(covs_iapf[:,i,i]), edgecolor=(0.2 , 0.8, 0.8, 0.9) , facecolor=(0.2, 0.8, 0.8, 0.3), label="std_iapf")
# plt.plot(mean_apf[:,i],'y',label='mean_apf')
# plt.fill_between(np.arange(len(mean_apf[:,i])), mean_apf[:,i] - np.sqrt(covs_apf[:,i,i]), mean_apf[:,i] + np.sqrt(covs_apf[:,i,i]), edgecolor=(1, 1, .4, 0.99) , facecolor=(1, 1, .4, 0.3), label="std_apf", linewidth=1.5)
# plt.legend()
# plt.show()









