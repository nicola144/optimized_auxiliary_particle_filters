import matplotlib.pyplot as plt
import numpy as np
from particles import *
from pykalman import KalmanFilter
from utils import *

dim = 2

n_particle = 10000


set_plotting()

# for more plots
# fig, ax = plt.subplots(1, dim)
# ax = ax.reshape(1, dim)
# fig.tight_layout(pad=0.3)
# plt.subplots_adjust(top=0.95)
# plt.tight_layout(pad=0.3)

# specify parameters

n_timesteps = 1

# transition_matrix = random_state.randn(2, 2).T.dot(random_state.randn(2, 2)) + 5.
# transition_offset = [0., 0.]
# observation_matrix =  random_state.randn(2, 2).T.dot(random_state.randn(2, 2))  + 7.  #np.eye(2) * 0.7
# observation_offset =  [0., 0.]
# transition_covariance = np.eye(2) * 20. #was it 10 ? 15 ? 5 ? 
# observation_covariance = np.eye(2) * 0.05
# initial_state_mean = [0., 0.]
# initial_state_covariance = np.eye(2)

seeds = np.loadtxt('seeds.out').astype('int64')


# transition_matrix = [[1, 0.1], [0, 1]]
# transition_offset = [-0.1, 0.1]
# observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
# observation_offset = [1.0, -1.0]
transition_matrix = np.eye(dim) * 1/2
observation_matrix = np.eye(dim) * 1/2
transition_offset = np.array([-2,2,-2,2,-2,2,-2,2,-2,2])
observation_offset = np.array([-2,2,-2,2,-2,2,-2,2,-2,2])

# transition_offset = np.array([-2,2,-2,2,-2])
# observation_offset = np.array([-2,2,-2,2,-2])
#
#
# transition_offset = np.array([-2,2])
# observation_offset = np.array([-2,2])


trans_var = 5.
obs_var = 2.5
transition_covariance = np.eye(dim) * trans_var
observation_covariance = np.eye(dim) * obs_var

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

assert is_pos_def(transition_covariance)
assert is_pos_def(observation_covariance)
assert is_pos_def(initial_state_covariance)

all_mean_deviations_bpf = []
all_mean_deviations_apf = []
all_mean_deviations_iapf = []
all_mean_deviations_oapf = []

all_logliks_deviations_bpf = []
all_logliks_deviations_apf = []
all_logliks_deviations_iapf = []
all_logliks_deviations_oapf = []

all_joint_logliks_deviations_bpf = []
all_joint_logliks_deviations_apf = []
all_joint_logliks_deviations_iapf = []
all_joint_logliks_deviations_oapf = []



for seed in tqdm(seeds):

    random_state = np.random.RandomState(seed)

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

    bpf = LinearGaussianBPF(
        init_particle=random_state.multivariate_normal(mean=initial_state_mean, cov=initial_state_covariance,
                                                       size=n_particle),
        random_state=random_state,
        transition_cov=transition_covariance,
        observation_cov=observation_covariance,
        transition_mat=transition_matrix,
        observation_mat=observation_matrix,
        transition_offset=transition_offset,
        observation_offset=observation_offset)

    apf = LinearGaussianAPF(
        init_particle=random_state.multivariate_normal(mean=initial_state_mean, cov=initial_state_covariance,
                                                       size=n_particle),
        random_state=random_state,
        transition_cov=transition_covariance,
        observation_cov=observation_covariance,
        transition_mat=transition_matrix,
        observation_mat=observation_matrix,
        transition_offset=transition_offset,
        observation_offset=observation_offset)

    iapf = LinearGaussianIAPF(
        init_particle=random_state.multivariate_normal(mean=initial_state_mean, cov=initial_state_covariance,
                                                       size=n_particle),
        random_state=random_state,
        transition_cov=transition_covariance,
        observation_cov=observation_covariance,
        transition_mat=transition_matrix,
        observation_mat=observation_matrix,
        transition_offset=transition_offset,
        observation_offset=observation_offset)

    oapf = LinearGaussianOAPF(
        init_particle=random_state.multivariate_normal(mean=initial_state_mean, cov=initial_state_covariance,
                                                       size=n_particle),
        random_state=random_state,
        transition_cov=transition_covariance,
        observation_cov=observation_covariance,
        transition_mat=transition_matrix,
        observation_mat=observation_matrix,
        transition_offset=transition_offset,
        observation_offset=observation_offset)

    mean_bpf, covs_bpf,  ess_bpf, n_unique_bpf, w_vars_bpf, liks_bpf, joint_liks_bpf = bpf.filter(observations)
    mean_apf, covs_apf,  ess_apf, n_unique_apf, w_vars_apf, liks_apf, joint_liks_apf = apf.filter(observations)
    mean_iapf, covs_iapf,  ess_iapf, n_unique_iapf, w_vars_iapf, liks_iapf, joint_liks_iapf = iapf.filter(observations)
    mean_oapf, covs_npf,  ess_npf, n_unique_npf, w_vars_npf, liks_oapf, joint_liks_oapf = oapf.filter(observations)

    sys.exit()


    # true results given by KF
    true_logliks, true_loglik = kf.loglikelihood(observations)
    joint_true_logliks = np.cumsum(true_logliks)

    mean_kf, covs_kf = kf.filter(observations)
    # mean_kf = np.zeros((n_timesteps, dim))
    # covs_kf = np.zeros((n_timesteps, dim, dim))
    mean_kf_estimate = np.zeros((n_timesteps, dim))

    # for t in range(n_timesteps - 1):
    #     if t == 0:
    #         mean_kf[t] = initial_state_mean
    #         covs_kf[t] = initial_state_covariance
    #         mean_kf_estimate[t] = np.average(random_state.multivariate_normal(mean=mean_kf[t],cov=covs_kf[t],size=n_particle),axis=0)
    #
    #     mean_kf[t + 1], covs_kf[t + 1] = (
    #         kf.filter_update(
    #             mean_kf[t],
    #             covs_kf[t],
    #             observations[t + 1],
    #             transition_offset=transition_offset,
    #         )
    #     )
    #     mean_kf_estimate[t+1] = np.average(random_state.multivariate_normal(mean=mean_kf[t + 1], cov=covs_kf[t + 1],size=n_particle))
    #     mean_lik = np.matmul(states[t], observation_matrix) + observation_offset
    #
    #     samples_posterior = random_state.multivariate_normal(mean=mean_kf[t+1],cov=covs_kf[t+1],size=n_particle)
    #
    #     mean_all = np.matmul(np.array(samples_posterior), observation_matrix) + observation_offset
    #     obs = torch.from_numpy(observations[t+1]).double()
    #     obs_all = obs[None, ...].repeat_interleave(n_particle, 0)
    #     mean_all = torch.from_numpy(mean_all).double()
    #     obs_cov = torch.from_numpy(observation_covariance).double()
    #     log_liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)
    #     log_prior = MultivariateNormal(torch.from_numpy(mean_kf[t]),torch.from_numpy(covs_kf[t])).log_prob(torch.from_numpy(samples_posterior))
    #
    #     inverse_constant = log_liks + log_prior
    #     constant_estimate =  logsumexp(inverse_constant, axis=0) - np.log(n_particle)
    #     if t==20:
    #         print(constant_estimate)
    #         print(true_logliks[20])
    #         sys.exit()


    #MEANS
    mean_deviations_bpf = np.average(mse(mean_bpf, mean_kf))
    mean_deviations_apf = np.average(mse(mean_apf, mean_kf))
    mean_deviations_iapf = np.average(mse(mean_iapf, mean_kf))
    mean_deviations_oapf = np.average(mse(mean_oapf, mean_kf))

    all_mean_deviations_bpf.append(mean_deviations_bpf)
    all_mean_deviations_apf.append(mean_deviations_apf)
    all_mean_deviations_iapf.append(mean_deviations_iapf)
    all_mean_deviations_oapf.append(mean_deviations_oapf)

    #partial constants
    logliks_deviations_bpf = np.average(mse(liks_bpf, true_logliks))
    logliks_deviations_apf = np.average(mse(liks_apf, true_logliks))
    logliks_deviations_iapf = np.average(mse(liks_iapf, true_logliks))
    logliks_deviations_oapf = np.average(mse(liks_oapf, true_logliks))
    all_logliks_deviations_bpf.append(logliks_deviations_bpf)
    all_logliks_deviations_apf.append(logliks_deviations_apf)
    all_logliks_deviations_iapf.append(logliks_deviations_iapf)
    all_logliks_deviations_oapf.append(logliks_deviations_oapf)

    #joint constants
    joint_logliks_deviations_bpf = np.average(mse(joint_liks_bpf, joint_true_logliks))
    joint_logliks_deviations_apf = np.average(mse(joint_liks_apf, joint_true_logliks))
    joint_logliks_deviations_iapf = np.average(mse(joint_liks_iapf, joint_true_logliks))
    joint_logliks_deviations_oapf = np.average(mse(joint_liks_oapf, joint_true_logliks))
    all_joint_logliks_deviations_bpf.append(joint_logliks_deviations_bpf)
    all_joint_logliks_deviations_apf.append(joint_logliks_deviations_apf)
    all_joint_logliks_deviations_iapf.append(joint_logliks_deviations_iapf)
    all_joint_logliks_deviations_oapf.append(joint_logliks_deviations_oapf)

    # plt.plot(liks_bpf, 'b', label='bpf')
    # plt.plot(liks_apf, 'y', label='apf')
    # plt.plot(liks_iapf, 'c', label='iapf')
    # plt.plot(liks_oapf, 'm', label='oapf')
    # plt.plot(true_logliks, 'r', label='Kalman F')
    # plt.title('tracking log Z')
    # plt.xlabel('Timstep')
    # plt.ylabel('log Z estimate')
    # plt.legend()
    # plt.show()
    # sys.exit()

res_means = np.vstack([
    all_mean_deviations_bpf,
    all_mean_deviations_apf,
    all_mean_deviations_iapf,
    all_mean_deviations_oapf
])

# res_logliks = np.vstack([
#     all_logliks_deviations_bpf,
#     all_logliks_deviations_apf,
#     all_logliks_deviations_iapf,
#     all_logliks_deviations_oapf
# ])

res_joint_logliks = np.vstack([
    all_joint_logliks_deviations_bpf,
    all_joint_logliks_deviations_apf,
    all_joint_logliks_deviations_iapf,
    all_joint_logliks_deviations_oapf
])



# REDUCED
# np.savetxt('results/lingauss/means/results_lingauss_'+str(n_particle)+'_reduced5_particles-dim'+str(dim)+'-trvar'+str(trans_var)+'-obsvar'+str(obs_var)+'.out', res_means, delimiter=',')
# np.savetxt('results/lingauss/logliks/results_lingauss_'+str(n_particle)+'_reduced5_particles-dim'+str(dim)+'-trvar'+str(trans_var)+'-obsvar'+str(obs_var)+'.out', res_logliks, delimiter=',')
np.savetxt('results/lingauss/joint_logliks/results_lingauss_'+str(n_particle)+'_reduced5_particles-dim'+str(dim)+'-trvar'+str(trans_var)+'-obsvar'+str(obs_var)+'.out', res_joint_logliks, delimiter=',')

# NONREDUCED
# np.savetxt('results/results_lingauss_'+str(n_particle)+'_particles-dim'+str(dim)+'.out', res, delimiter=',')



    # print('-----------------------\n')
    # print("MSE mean")
    # print(np.average(mse(mean_bpf, mean_kf)))
    # print(np.average(mse(mean_apf, mean_kf)))
    # print(np.average(mse(mean_iapf, mean_kf)))
    # print(np.average(mse(mean_npf, mean_kf)))
    # print('-----------------------\n')
    # print('-----------------------\n')

    # lik_bpf = np.sum(liks_bpf) - (n_timesteps * np.log(n_particle_bpf))
    # lik_apf = np.sum(liks_apf) - (n_timesteps * np.log(n_particle_bpf))
    # lik_iapf = np.sum(liks_iapf) - (n_timesteps * np.log(n_particle_bpf))

    # print('true ', true_loglik)
    #
    # print('bpf ',lik_bpf)
    # print('apf ', lik_apf)
    # print('iapf', lik_iapf)

    # print('-----------------------\n')

    # print("MSE LOGZ ")
    # print((lik_bpf - true_loglik)**2)
    # print((lik_apf - true_loglik)**2)
    # print((lik_iapf - true_loglik)**2)

    # print(np.average(mse(liks_npf,true_logliks)))
    # print('-----------------------\n')

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
    # # plt.plot(true_logliks, 'r', label='Kalman F')
    # plt.title('tracking log Z')
    # plt.xlabel('Timstep')
    # plt.ylabel('log Z estimate')
    # plt.legend()
    # plt.show()


    # for row in ax:
    #     for i,col in enumerate(row):
    #
    #         # lines_true = col.plot(states[:,i], '*--', color='k', label='true')
    #
    #         lines_filt_kf = col.plot(mean_kf[:,i], linewidth=2 ,color='r',label='mean_kf')
    #
    #         lines_filt_bpf = col.plot(mean_bpf[:,i],  'o--' , linewidth=1.2 ,color='b',label='mean_bpf',markersize=3)
    #
    #         lines_filt_apf = col.plot(mean_apf[:,i], 'v--', linewidth=1.2,color='y',label='mean_apf',markersize=3)
    #
    #         lines_filt_iapf = col.plot(mean_iapf[:,i], '2--', linewidth=1.2 ,color='c',label='mean_iapf',markersize=3)
    #
    #         lines_filt_npf = col.plot(mean_npf[:,i], 'D--' , linewidth=1.2 ,color='m',label='mean_oapf',markersize=3)
    #
    #         # lines_smooth = plt.plot(smoothed_state_estimates, color='g')
    #
    #         # col.fill_between(np.arange(len(mean_npf[:,i])), mean_npf[:,i] - np.sqrt(covs_npf[:,i,i]), mean_npf[:,i] + np.sqrt(covs_npf[:,i,i]), edgecolor=(1 , 0.2, 0.8, 0.99) , facecolor=(1, 0.2, 0.8, 0.3), label="std_oapf", linewidth=1.5)
    #         # col.fill_between(np.arange(len(mean_bpf[:,i])), mean_bpf[:,i] - np.sqrt(covs_bpf[:,i,i]), mean_bpf[:,i] + np.sqrt(covs_bpf[:,i,i]), edgecolor=(0 , 0, 1, 0.99) , facecolor=(0, 0, 1, 0.3), label="std_bpf", linewidth=1)
    #         # col.fill_between(np.arange(len(mean_apf[:,i])), mean_apf[:,i] - np.sqrt(covs_apf[:,i,i]), mean_apf[:,i] + np.sqrt(covs_apf[:,i,i]), edgecolor=(0 , 1, 1, 0.99) , facecolor=(0, 1, 1, 0.3), label="std_apf")
    #         # col.fill_between(np.arange(len(mean_iapf[:,i])), mean_iapf[:,i] - np.sqrt(covs_iapf[:,i,i]), mean_iapf[:,i] + np.sqrt(covs_iapf[:,i,i]), edgecolor=(0.2 , 0.8, 0.8, 0.9) , facecolor=(0.2, 0.8, 0.8, 0.3), label="std_iapf")
    #         # col.fill_between(np.arange(len(mean_kf[:,i])), mean_kf[:,i] - np.sqrt(covs_kf[:,i,i]), mean_kf[:,i] + np.sqrt(covs_kf[:,i,i]), edgecolor=(1, 0, 0, 0.9), facecolor=(1, 0, 0, 0.3),  label="std_kf", linewidth=1.5)
    #
    #         obs = observations[:,i] - observation_offset[i]
    #         col.scatter(np.arange(n_timesteps), obs, s=50, facecolors='none', edgecolors='g', label='obs', linewidth=1)
    #         col.legend()
    #         col.yaxis.zoom(0.3)
    #
    #
    # plt.xlabel('timestep')
    # plt.ylabel('hidden state')
    #
    # # plt.show()
    # plt.savefig('imgs/mean-paper.pdf',bbox_inches='tight')