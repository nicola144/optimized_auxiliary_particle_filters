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

n_particle_bpf = 100
n_particle_iapf = 100
n_particle_npf = 100



set_plotting()

# for more plots
# fig, ax = plt.subplots(1, dim)
# ax = ax.reshape(1,dim)
# fig.tight_layout(pad=0.3)
# plt.subplots_adjust(top=0.95)
plt.tight_layout(pad=0.3)

# specify parameters


n_timesteps = 100

# transition_matrix = random_state.randn(2, 2).T.dot(random_state.randn(2, 2)) + 5.
# transition_offset = [0., 0.]
# observation_matrix =  random_state.randn(2, 2).T.dot(random_state.randn(2, 2))  + 7.  #np.eye(2) * 0.7
# observation_offset =  [0., 0.]
# transition_covariance = np.eye(2) * 20. #was it 10 ? 15 ? 5 ? 
# observation_covariance = np.eye(2) * 0.05
# initial_state_mean = [0., 0.]
# initial_state_covariance = np.eye(2)

random_state = np.random.RandomState(random_seed)

# transition_matrix = [[1, 0.1], [0, 1]]
# transition_offset = [-0.1, 0.1]
# observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
# observation_offset = [1.0, -1.0]
transition_matrix = np.eye(dim)
observation_matrix = np.eye(dim)
transition_offset = np.zeros((dim,))
observation_offset = np.zeros((dim,))

transition_covariance = np.eye(dim) * 1.
observation_covariance = np.eye(dim) * 1.

# R = random_state.randint(low=0,high=3,size=(dim,dim)) + random_state.randn(dim, dim)
# transition_covariance = R.T.dot(R) + 5. * np.eye(dim)
# RR = random_state.randint(low=0,high=3,size=(dim,dim)) + random_state.randn(dim, dim)
# observation_covariance = RR.T.dot(RR) + 0.1 * np.eye(dim)

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

bpf = LinearGaussianBPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle_bpf),
						random_state=random_state,
						transition_cov=transition_covariance,
						observation_cov=observation_covariance,
						transition_mat=transition_matrix,
						observation_mat=observation_matrix,
						transition_offset=transition_offset,
						observation_offset=observation_offset )

apf = LinearGaussianAPF(init_particle=random_state.multivariate_normal(mean=initial_state_mean,cov=initial_state_covariance,size=n_particle_bpf),
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

preds_bpf, covs_bpf, liks_bpf, ess_bpf = bpf.filter(observations)
preds_apf, covs_apf, liks_apf, ess_apf = apf.filter(observations)
# preds_iapf, covs_iapf, liks_iapf, ess_iapf = iapf.filter(observations)
# preds_npf, covs_npf, liks_npf, ess_npf = npf.filter(observations)


# estimate state with filtering and smoothing
preds_kf, covs_kf = kf.filter(observations)
true_logliks = kf.loglikelihood(observations)[0]


print('-----------------------\n')
print("MSE mean")
print(np.average(mse(preds_bpf,preds_kf)))
print(np.average(mse(preds_apf,preds_kf)))
# print(np.average(mse(preds_iapf,preds_kf)))
# print(np.average(mse(preds_npf,preds_kf)))
print('-----------------------\n')
print('-----------------------\n')
print("MSE LOGZ ")
print(np.average(mse(liks_bpf,true_logliks)))
print(np.average(mse(liks_apf,true_logliks)))
# print(np.average(mse(liks_iapf,true_logliks)))
# print(np.average(mse(liks_npf,true_logliks)))
print('-----------------------\n')

# plt.plot( ess_bpf, 'b')
# plt.plot(ess_apf, 'y')
# plt.plot(ess_iapf, 'c')
# plt.plot(ess_npf, 'm')
# plt.show()

# plt.clf()

plt.plot( liks_bpf, 'b', label='bpf')
plt.plot(liks_apf, 'y', label='apf')
# plt.plot(liks_iapf, 'c', label='iapf')
# plt.plot(liks_npf, 'm', label='new pf')
plt.plot(true_logliks, 'r', label='Kalman F')
plt.title('tracking log Z')
plt.xlabel('Timstep')
plt.ylabel('log Z estimate')
plt.legend()
plt.show()

sys.exit()

# draw estimates
for row in ax:
    for i,col in enumerate(row):

        # lines_true = col.plot(states[:,i], '*--', color='k', label='true')

        lines_filt_kf = col.plot(preds_kf[:,i], linewidth=4 ,color='r',label='pred_kf')

        lines_filt_bpf = col.plot(preds_bpf[:,i], 'o--' ,color='b',label='pred_bpf')

        lines_filt_apf = col.plot(preds_apf[:,i], 'v--',color='y',label='pred_apf')

        lines_filt_iapf = col.plot(preds_iapf[:,i], '2--',color='c',label='pred_iapf')

        lines_filt_npf = col.plot(preds_npf[:,i], 'D--' ,color='m',label='pred_npf')

        # lines_smooth = plt.plot(smoothed_state_estimates, color='g')

        # col.fill_between(np.arange(len(preds_npf[:,i])), preds_npf[:,i] - np.sqrt(covs_npf[:,i,i]), preds_npf[:,i] + np.sqrt(covs_npf[:,i,i]), color='m', alpha=0.5, label="std_npf")
        # col.fill_between(np.arange(len(preds_bpf[:,i])), preds_bpf[:,i] - np.sqrt(covs_bpf[:,i,i]), preds_bpf[:,i] + np.sqrt(covs_bpf[:,i,i]), color='b', alpha=0.5, label="std_bpf")
        # col.fill_between(np.arange(len(preds_apf[:,i])), preds_apf[:,i] - np.sqrt(covs_apf[:,i,i]), preds_apf[:,i] + np.sqrt(covs_apf[:,i,i]), color='y', alpha=0.5, label="std_apf")
        # col.fill_between(np.arange(len(preds_iapf[:,i])), preds_iapf[:,i] - np.sqrt(covs_iapf[:,i,i]), preds_iapf[:,i] + np.sqrt(covs_iapf[:,i,i]), color='c', alpha=0.5, label="std_iapf")
        # col.fill_between(np.arange(len(preds_kf[:,i])), preds_kf[:,i] - np.sqrt(covs_kf[:,i,i]), preds_kf[:,i] + np.sqrt(covs_kf[:,i,i]), color='r', alpha=0.5, label="std_kf")

        obs = observations[:,i] - observation_offset[i]
        col.scatter(np.arange(n_timesteps), obs, s=60, facecolors='none', edgecolors='g', label='obs')
        col.legend()

plt.xlabel('timestep')
plt.ylabel('hidden state')

plt.show()
