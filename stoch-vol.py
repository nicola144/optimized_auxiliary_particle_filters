

from utils import *
from particles import *

random_state = np.random.RandomState(random_seed)

dim = 2
timesteps = 100
n_particle_bpf = 100000
n_particle_apf = 100
n_particle_iapf = 100
n_particle_npf = 100


constant_mean = np.zeros(dim,)
initial_cov = np.eye(dim)
transition_cov = 0.1 * np.eye(dim)

# phi = np.diag( random_state.randint(low=0,high=5,size=dim ) )
phi = np.diag(np.ones(dim,))

def prior():
	return random_state.multivariate_normal(mean=constant_mean,cov=initial_cov)

def transition(prev_state):
	return random_state.multivariate_normal(mean=constant_mean + np.dot(phi, prev_state - constant_mean), cov=transition_cov)

def observation(curr_state):
	obs_cov = np.diag(np.exp(curr_state))
	assert np.all( np.linalg.eigh( obs_cov )[0] > 0 )
	return random_state.multivariate_normal(mean=np.zeros(dim,), cov=obs_cov) 


obs = []

curr_state = observation(prior())

obs.append( curr_state )

for t in range(timesteps - 1):
	# transition state
	curr_state = transition(curr_state)

	#get obs
	obs.append(observation(curr_state))


observations = np.array(obs)



bpf = LinearGaussianBPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle_bpf),
						random_state=random_state,
						transition_cov=transition_covariance,
						observation_cov=observation_covariance,
						transition_mat=transition_matrix,
						observation_mat=observation_matrix,
						transition_offset=transition_offset,
						observation_offset=observation_offset )

apf = LinearGaussianAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle_apf),
                        random_state=random_state,
                        transition_cov=transition_covariance,
                        observation_cov=observation_covariance,
                        transition_mat=transition_matrix,
                        observation_mat=observation_matrix,
                        transition_offset=transition_offset,
                        observation_offset=observation_offset )

iapf = LinearGaussianIAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle_iapf),
                        random_state=random_state,
                        transition_cov=transition_covariance,
                        observation_cov=observation_covariance,
                        transition_mat=transition_matrix,
                        observation_mat=observation_matrix,
                        transition_offset=transition_offset,
                        observation_offset=observation_offset )

npf = LinearGaussianNewAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle_npf),
                        random_state=random_state,
                        transition_cov=transition_covariance,
                        observation_cov=observation_covariance,
                        transition_mat=transition_matrix,
                        observation_mat=observation_matrix,
                        transition_offset=transition_offset,
                        observation_offset=observation_offset )

mean_bpf, covs_bpf, liks_bpf, ess_bpf = bpf.filter(observations)
mean_apf, covs_apf, liks_apf, ess_apf= apf.filter(observations)
mean_iapf, covs_iapf, liks_iapf, ess_iapf = iapf.filter(observations)
mean_npf, covs_npf, liks_npf, ess_npf = npf.filter(observations)


print('-----------------------\n')
print("MSE mean")
print(np.average(mse(mean_apf,mean_bpf)))
print(np.average(mse(mean_iapf,mean_bpf)))
print(np.average(mse(mean_npf,mean_bpf)))
print('-----------------------\n')
print('-----------------------\n')



