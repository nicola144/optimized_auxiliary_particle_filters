import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm

from pykalman import KalmanFilter
from scipy.linalg import eigh

from particles import *
from utils import *
import sys

dim = 2
n_particle = 100

set_plotting()

fig, ax = plt.subplots(1, dim)
ax = ax.reshape(1,dim)

fig.tight_layout(pad=0.3)
plt.subplots_adjust(top=0.95)


# specify parameters


n_timesteps = 50

# transition_matrix = random_state.randn(2, 2).T.dot(random_state.randn(2, 2)) + 5.
# transition_offset = [0., 0.]
# observation_matrix =  random_state.randn(2, 2).T.dot(random_state.randn(2, 2))  + 7.  #np.eye(2) * 0.7
# observation_offset =  [0., 0.]
# transition_covariance = np.eye(2) * 20. #was it 10 ? 15 ? 5 ? 
# observation_covariance = np.eye(2) * 0.05
# initial_state_mean = [0., 0.]
# initial_state_covariance = np.eye(2)

random_state = np.random.RandomState(1)

# transition_matrix = [[1, 0.1], [0, 1]]
# transition_offset = [-0.1, 0.1]
# observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
# observation_offset = [1.0, -1.0]
transition_matrix = np.eye(dim)
observation_matrix = np.eye(dim)
transition_offset = np.zeros((dim,))
observation_offset = np.zeros((dim,))

transition_covariance = np.eye(dim) * 1.
observation_covariance = np.eye(dim) * 0.095

# s1 = random_state.randn(2, 2)
# s2 = random_state.randn(2, 2)
# transition_covariance = s1.T.dot(s1) + random_state.randint(low=-5,high=5,size=(2,2))
# observation_covariance = s2.T.dot(s2) + random_state.randint(low=-1,high=1,size=(2,2))


# initial_state_mean = [5, -5]
# initial_state_covariance = [[1, 0.1], [-0.1, 1]]
initial_state_mean = np.zeros((dim,))
initial_state_covariance = np.eye(dim)


assert np.all( np.linalg.eigh(transition_covariance)[0] > 0 )
assert np.all( np.linalg.eigh(observation_covariance)[0] > 0 )
assert np.all( np.linalg.eigh(initial_state_covariance)[0] > 0 )


# sample from model
kf = KalmanFilter(
    transition_matrix, observation_matrix, transition_covariance,
    observation_covariance, transition_offset, observation_offset,
    initial_state_mean, initial_state_covariance,
    random_state=random_state
)
states, observations = kf.sample(
    n_timesteps=n_timesteps,
    initial_state=initial_state_mean
)

bpf = LinearGaussianBPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle),
						random_state=random_state,
						transition_cov=transition_covariance,
						observation_cov=observation_covariance,
						transition_mat=transition_matrix,
						observation_mat=observation_matrix,
						transition_offset=transition_offset,
						observation_offset=observation_offset )

apf = LinearGaussianAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle),
                        random_state=random_state,
                        transition_cov=transition_covariance,
                        observation_cov=observation_covariance,
                        transition_mat=transition_matrix,
                        observation_mat=observation_matrix,
                        transition_offset=transition_offset,
                        observation_offset=observation_offset )

iapf = LinearGaussianIAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle),
                        random_state=random_state,
                        transition_cov=transition_covariance,
                        observation_cov=observation_covariance,
                        transition_mat=transition_matrix,
                        observation_mat=observation_matrix,
                        transition_offset=transition_offset,
                        observation_offset=observation_offset )

npf = LinearGaussianNewAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle),
                        random_state=random_state,
                        transition_cov=transition_covariance,
                        observation_cov=observation_covariance,
                        transition_mat=transition_matrix,
                        observation_mat=observation_matrix,
                        transition_offset=transition_offset,
                        observation_offset=observation_offset )

preds_bpf, covs_bpf = bpf.filter(observations)
preds_apf, covs_apf = apf.filter(observations)
preds_iapf, covs_iapf = iapf.filter(observations)
preds_npf, covs_npf = npf.filter(observations)


# estimate state with filtering and smoothing
preds_kf, covs_kf = kf.filter(observations)
# smoothed_state_estimates, smoothed_covariances = kf.smooth(observations)

print('-----------------------\n')
print(np.average(mse(preds_bpf,preds_kf)))
print(np.average(mse(preds_apf,preds_kf)))
print(np.average(mse(preds_iapf,preds_kf)))
print(np.average(mse(preds_npf,preds_kf)))
print('-----------------------\n')

# draw estimates
# for row in ax:
#     for i,col in enumerate(row):

#         # lines_true = col.plot(states[:,i], '*--', color='k', label='true')

#         lines_filt_kf = col.plot(preds_kf[:,i], '3--' ,color='r',label='pred_kf')

#         lines_filt_bpf = col.plot(preds_bpf[:,i], 'o--' ,color='b',label='pred_bpf')

#         lines_filt_apf = col.plot(preds_apf[:,i], 'v--',color='y',label='pred_apf')

#         lines_filt_iapf = col.plot(preds_iapf[:,i], '2--',color='c',label='pred_iapf')

#         lines_filt_npf = col.plot(preds_npf[:,i], 'D--' ,color='m',label='pred_npf')

#         # lines_smooth = plt.plot(smoothed_state_estimates, color='g')
#         # col.fill_between(np.arange(len(filtered_state_estimates[:,i])), filtered_state_estimates[:,i] - np.sqrt(filtered_covariances[:,i,i]), filtered_state_estimates[:,i] + np.sqrt(filtered_covariances[:,i,i]), color="orange", alpha=0.5, label="std_pred")
#         obs = observations[:,i] - observation_offset[i]
#         col.scatter(np.arange(n_timesteps), obs, s=60, facecolors='none', edgecolors='g', label='obs')
#         col.legend()
# # plt.savefig('kf.png')
# plt.show()
