

from utils import *

random_state = np.random.RandomState(random_seed)

dim = 2
timesteps = 100

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


obs = np.array(obs)
plt.plot(range(timesteps), obs[:, 0])
plt.show()



