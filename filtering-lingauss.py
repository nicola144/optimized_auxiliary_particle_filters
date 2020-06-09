import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm

from pykalman import KalmanFilter
from scipy.linalg import eigh

from particles import *
import sys

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
        'legend.loc': 'lower right'
        }
rcParams['agg.path.chunksize'] = 10000
rcParams.update(params)
plt.style.use('bmh')


# specify parameters
random_state = np.random.RandomState(0)
transition_matrix = np.array([[1, 0.1], [0, 1]])
transition_offset = np.array([-0.1, 0.1])
observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
observation_offset = np.array([1.0, -1.0])
transition_covariance = np.eye(2) 
# observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1  # original
observation_covariance = np.eye(2) * 0.1
initial_state_mean = np.array([5, -5])
initial_state_covariance = np.array([[1, 0.1], [-0.1, 1]])


assert np.all( np.linalg.eigh(transition_covariance)[0] > 0)
assert np.all( np.linalg.eigh(observation_covariance)[0] > 0)
assert np.all( np.linalg.eigh(initial_state_covariance)[0] > 0)


# sample from model
kf = KalmanFilter(
    transition_matrix, observation_matrix, transition_covariance,
    observation_covariance, transition_offset, observation_offset,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)
states, observations = kf.sample(
    n_timesteps=50,
    initial_state=initial_state_mean
)

bpf = LinearGaussianBPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=100),
						random_state=random_state,
						transition_cov=transition_covariance,
						observation_cov=observation_covariance,
						transition_mat=transition_matrix,
						observation_mat=observation_matrix,
						transition_offset=transition_offset,
						observation_offset=observation_offset )

filtered_state_estimates, filtered_covariances = bpf.filter(observations)


# estimate state with filtering and smoothing
# filtered_state_estimates, filtered_covariances = kf.filter(observations)
# smoothed_state_estimates, smoothed_covariances = kf.smooth(observations)

print(np.average((filtered_state_estimates - states)**2, axis=0))
# sys.exit()

# draw estimates
plt.figure()
lines_true = plt.plot(states[:,0], color='b')
lines_filt = plt.plot(filtered_state_estimates[:,0], color='r')
# lines_smooth = plt.plot(smoothed_state_estimates, color='g')
cov = plt.fill_between(np.arange(len(filtered_state_estimates[:,0])), filtered_state_estimates[:,0] - np.sqrt(filtered_covariances[:,0,0]), filtered_state_estimates[:,0] + np.sqrt(filtered_covariances[:,0,0]), color="orange", alpha=0.5, label="filt_std")
plt.legend((lines_true[0], lines_filt[0], cov), ('true', 'filtered', 'filt_std'))

plt.show()