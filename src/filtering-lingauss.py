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

n_particle_bpf = 10
n_particle_iapf = 10
n_particle_npf = 10



set_plotting()

# for more plots
fig, ax = plt.subplots(1, dim)
ax = ax.reshape(1,dim)
fig.tight_layout(pad=0.3)
plt.subplots_adjust(top=0.95)
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

transition_covariance = np.eye(dim) * 5.
observation_covariance = np.eye(dim) * .2

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

mean_bpf, covs_bpf, liks_bpf, ess_bpf, n_unique_bpf, w_vars_bpf = bpf.filter(observations)
mean_apf, covs_apf, liks_apf, ess_apf, n_unique_apf, w_vars_apf = apf.filter(observations)
mean_iapf, covs_iapf, liks_iapf, ess_iapf, n_unique_iapf, w_vars_iapf  = iapf.filter(observations)
mean_npf, covs_npf, liks_npf, ess_npf, n_unique_npf, w_vars_npf  = npf.filter(observations)


# estimate state with filtering and smoothing
mean_kf, covs_kf = kf.filter(observations)
true_logliks,true_loglik = kf.loglikelihood(observations)

print('-----------------------\n')
print("MSE mean")
print(np.average(mse(mean_bpf,mean_kf)))
print(np.average(mse(mean_apf,mean_kf)))
print(np.average(mse(mean_iapf,mean_kf)))
print(np.average(mse(mean_npf,mean_kf)))
print('-----------------------\n')
print('-----------------------\n')


# lik_bpf = np.sum(liks_bpf)
# lik_apf = np.sum(liks_apf)
# lik_iapf = np.sum(liks_iapf)

# print('true ', true_loglik)

# print('bpf ',lik_bpf)
# print('apf ', lik_apf)
# print('iapf', lik_iapf)




# print("MSE LOGZ ")
# print(mse(lik_bpf,true_loglik))
# print(mse(lik_apf,true_loglik))
# print(mse(lik_iapf,true_loglik))
# print((lik_bpf - true_loglik)**2)
# print((lik_apf - true_loglik)**2)
# print((lik_iapf - true_loglik)**2)

# print(np.average(mse(liks_npf,true_logliks)))
print('-----------------------\n')

# plt.plot(ess_bpf, 'b', label='bpf')
# plt.plot(ess_apf, 'y', label='apf')
# plt.plot(ess_iapf, 'c', label='iapf')
# plt.plot(ess_npf, 'm', label='npf')
# plt.xlabel('timestep')
# plt.ylabel('ess')
# plt.savefig('imgs/ess2.pdf', bbox_inches='tight')
# sys.exit()


# plt.plot( liks_bpf, 'b', label='bpf')
# plt.plot(liks_apf, 'y', label='apf')
# plt.plot(liks_iapf, 'c', label='iapf')
# # plt.plot(liks_npf, 'm', label='new pf')
# plt.plot(true_logliks, 'r', label='Kalman F')
# plt.title('tracking log Z')
# plt.xlabel('Timstep')
# plt.ylabel('log Z estimate')
# plt.legend()
# plt.show()
# sys.exit()





for row in ax:
    for i,col in enumerate(row):

        # lines_true = col.plot(states[:,i], '*--', color='k', label='true')

        lines_filt_kf = col.plot(mean_kf[:,i], linewidth=2 ,color='r',label='mean_kf')

        lines_filt_bpf = col.plot(mean_bpf[:,i],  'o--' , linewidth=1.2 ,color='b',label='mean_bpf',markersize=3)

        lines_filt_apf = col.plot(mean_apf[:,i], 'v--', linewidth=1.2,color='y',label='mean_apf',markersize=3)

        lines_filt_iapf = col.plot(mean_iapf[:,i], '2--', linewidth=1.2 ,color='c',label='mean_iapf',markersize=3)

        lines_filt_npf = col.plot(mean_npf[:,i], 'D--' , linewidth=1.2 ,color='m',label='mean_npf',markersize=3)

        # lines_smooth = plt.plot(smoothed_state_estimates, color='g')

        # col.fill_between(np.arange(len(mean_npf[:,i])), mean_npf[:,i] - np.sqrt(covs_npf[:,i,i]), mean_npf[:,i] + np.sqrt(covs_npf[:,i,i]), edgecolor=(1 , 0.2, 0.8, 0.99) , facecolor=(1, 0.2, 0.8, 0.3), label="std_npf", linewidth=1.5)
        # col.fill_between(np.arange(len(mean_bpf[:,i])), mean_bpf[:,i] - np.sqrt(covs_bpf[:,i,i]), mean_bpf[:,i] + np.sqrt(covs_bpf[:,i,i]), edgecolor=(0 , 0, 1, 0.99) , facecolor=(0, 0, 1, 0.3), label="std_bpf", linewidth=1)
        # col.fill_between(np.arange(len(mean_apf[:,i])), mean_apf[:,i] - np.sqrt(covs_apf[:,i,i]), mean_apf[:,i] + np.sqrt(covs_apf[:,i,i]), edgecolor=(0 , 1, 1, 0.99) , facecolor=(0, 1, 1, 0.3), label="std_apf")
        # col.fill_between(np.arange(len(mean_iapf[:,i])), mean_iapf[:,i] - np.sqrt(covs_iapf[:,i,i]), mean_iapf[:,i] + np.sqrt(covs_iapf[:,i,i]), edgecolor=(0.2 , 0.8, 0.8, 0.9) , facecolor=(0.2, 0.8, 0.8, 0.3), label="std_iapf")
        # col.fill_between(np.arange(len(mean_kf[:,i])), mean_kf[:,i] - np.sqrt(covs_kf[:,i,i]), mean_kf[:,i] + np.sqrt(covs_kf[:,i,i]), edgecolor=(1, 0, 0, 0.9), facecolor=(1, 0, 0, 0.3),  label="std_kf", linewidth=1.5)

        obs = observations[:,i] - observation_offset[i]
        col.scatter(np.arange(n_timesteps), obs, s=50, facecolors='none', edgecolors='g', label='obs', linewidth=1)
        col.legend()

plt.xlabel('timestep')
plt.ylabel('hidden state')

plt.savefig('imgs/means.pdf', bbox_inches='tight')


