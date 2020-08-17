

from utils import *
from particles import *

random_state = np.random.RandomState(random_seed)

dim = 2
timesteps = 100
n_particle_bpf = 10000
n_particle_apf = 100
n_particle_iapf = 100
n_particle_npf = 100


constant_mean = np.zeros(dim,)
initial_cov = np.eye(dim)
transition_cov = 0.1 * np.eye(dim)

# phi = np.diag( random_state.randint(low=0,high=5,size=dim ) )
phi = np.diag(np.ones(dim,))

def prior_sample(size=1):
	res = random_state.multivariate_normal(mean=constant_mean,cov=initial_cov,size=size)
	return res.squeeze()

def transition_sample(prev_state):
	return random_state.multivariate_normal(mean=constant_mean + np.dot(phi, prev_state - constant_mean), cov=transition_cov)

def observation_sample(curr_state):
	obs_cov = np.diag(np.exp(curr_state))

	assert np.all( np.linalg.eigh( obs_cov )[0] > 0 )
	return random_state.multivariate_normal(mean=np.zeros(dim,), cov=obs_cov) 


obs = []

curr_state = observation_sample(prior_sample())

obs.append( curr_state )

for t in range(timesteps - 1):
	# transition state
	curr_state = transition_sample(curr_state)

	#get obs
	obs.append(observation_sample(curr_state))


observations = np.array(obs)

bpf = StochVolBPF(init_particle=prior_sample(size=n_particle_bpf),
						random_state=random_state,
						transition_cov=transition_cov,
						transition_offset=constant_mean,
						phi=phi )

# apf = LinearGaussianAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle_apf),
#                         random_state=random_state,
#                         transition_cov=transition_covariance,
#                         observation_cov=observation_covariance,
#                         transition_mat=transition_matrix,
#                         observation_mat=observation_matrix,
#                         transition_offset=transition_offset,
#                         observation_offset=observation_offset )

# iapf = LinearGaussianIAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle_iapf),
#                         random_state=random_state,
#                         transition_cov=transition_covariance,
#                         observation_cov=observation_covariance,
#                         transition_mat=transition_matrix,
#                         observation_mat=observation_matrix,
#                         transition_offset=transition_offset,
#                         observation_offset=observation_offset )

# npf = LinearGaussianNewAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle_npf),
#                         random_state=random_state,
#                         transition_cov=transition_covariance,
#                         observation_cov=observation_covariance,
#                         transition_mat=transition_matrix,
#                         observation_mat=observation_matrix,
#                         transition_offset=transition_offset,
#                         observation_offset=observation_offset )

mean_bpf, covs_bpf, liks_bpf, ess_bpf = bpf.filter(observations)

i = 0
plt.plot(observations[:,i],'r')
plt.plot(mean_bpf[:,i],'b')
plt.fill_between(np.arange(len(mean_bpf[:,i])), mean_bpf[:,i] - np.sqrt(covs_bpf[:,i,i]), mean_bpf[:,i] + np.sqrt(covs_bpf[:,i,i]), edgecolor=(0 , 0, 1, 0.99) , facecolor=(0, 0, 1, 0.3), label="std_bpf", linewidth=1)

plt.show()
sys.exit()
# mean_apf, covs_apf, liks_apf, ess_apf= apf.filter(observations)
# mean_iapf, covs_iapf, liks_iapf, ess_iapf = iapf.filter(observations)
# mean_npf, covs_npf, liks_npf, ess_npf = npf.filter(observations)


# print('-----------------------\n')
# print("MSE mean")
# print(np.average(mse(mean_apf,mean_bpf)))
# print(np.average(mse(mean_iapf,mean_bpf)))
# print(np.average(mse(mean_npf,mean_bpf)))
# print('-----------------------\n')
# print('-----------------------\n')



