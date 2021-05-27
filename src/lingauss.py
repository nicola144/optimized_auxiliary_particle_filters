import matplotlib.pyplot as plt
import numpy as np
from particles import *
from pykalman import KalmanFilter
from utils import *

# dim = 10


set_plotting()

# specify parameters

n_timesteps = 100

seeds = np.loadtxt('seeds.out').astype('int64')

transition_matrix = np.eye(dim) * 1/2
observation_matrix = np.eye(dim) * 1/2

transition_offset = np.array([0]*dim)
observation_offset = np.array([0]*dim)


trans_var = 2.5
obs_var = 5.
transition_covariance = np.eye(dim) * trans_var
observation_covariance = np.eye(dim) * obs_var

initial_state_mean = np.zeros((dim,))
initial_state_covariance = np.eye(dim)

assert is_pos_def(transition_covariance)
assert is_pos_def(observation_covariance)
assert is_pos_def(initial_state_covariance)

all_mean_deviations_bpf = []
all_mean_deviations_apf = []
all_mean_deviations_iapf = []
all_mean_deviations_oapf = []
all_mean_deviations_faapf = []
all_mean_deviations_approx_kf = []

all_joint_logliks_deviations_bpf = []
all_joint_logliks_deviations_apf = []
all_joint_logliks_deviations_iapf = []
all_joint_logliks_deviations_oapf = []
all_joint_logliks_deviations_faapf = []

for n_particle in tqdm([10,20,40,60,80,100,200,400,600,800,1000]):

    mean_deviations_bpf = []
    mean_deviations_apf = []
    mean_deviations_iapf = []
    mean_deviations_oapf = []
    mean_deviations_faapf = []
    mean_deviations_approx_kf = []

    joint_logliks_deviations_bpf = []
    joint_logliks_deviations_apf = []
    joint_logliks_deviations_iapf = []
    joint_logliks_deviations_oapf = []
    joint_logliks_deviations_faapf = []

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


        faapf = LinearGaussianFullyAdapted(
            init_particle=random_state.multivariate_normal(mean=initial_state_mean, cov=initial_state_covariance,
                                                           size=n_particle),
            random_state=random_state,
            transition_cov=transition_covariance,
            observation_cov=observation_covariance,
            transition_mat=transition_matrix,
            observation_mat=observation_matrix,
            transition_offset=transition_offset,
            observation_offset=observation_offset)


        mean_bpf, covs_bpf,  ess_bpf, n_unique_bpf, w_vars_bpf, liks_bpf, joint_liks_bpf, times_bpf = bpf.filter(observations)
        mean_apf, covs_apf,  ess_apf, n_unique_apf, w_vars_apf, liks_apf, joint_liks_apf, times_apf = apf.filter(observations)
        mean_iapf, covs_iapf,  ess_iapf, n_unique_iapf, w_vars_iapf, liks_iapf, joint_liks_iapf, times_iapf = iapf.filter(observations)
        mean_oapf, covs_npf,  ess_npf, n_unique_npf, w_vars_npf, liks_oapf, joint_liks_oapf, times_iapf = oapf.filter(observations)
        mean_faapf, covs_faapf,  ess_faapf, n_unique_faapf, w_vars_faapf, liks_faapf, joint_liks_faapf, times_faapf = faapf.filter(observations)


        # true results given by KF
        true_logliks, true_loglik = kf.loglikelihood(observations)
        joint_true_logliks = np.cumsum(true_logliks)

        mean_kf, covs_kf = kf.filter(observations)

        approximate_mean_kf = []
        for mean,cov in zip(mean_kf,covs_kf):
            samples = random_state.multivariate_normal(mean=mean, cov=cov, size=n_particle)
            # approximate KF mean
            approximate_mean_kf.append(np.average(samples,axis=0))
        approximate_mean_kf = np.asarray(approximate_mean_kf)

        #MEANS
        mean_deviations_bpf.append(np.average(mse(mean_bpf, mean_kf)))
        mean_deviations_apf.append(np.average(mse(mean_apf, mean_kf)))
        mean_deviations_iapf.append(np.average(mse(mean_iapf, mean_kf)))
        mean_deviations_oapf.append(np.average(mse(mean_oapf, mean_kf)))
        mean_deviations_faapf.append(np.average(mse(mean_faapf, mean_kf)))
        mean_deviations_approx_kf.append(np.average(mse(approximate_mean_kf, mean_kf)))

        #joint constants

        joint_logliks_deviations_bpf.append(np.average(mse(joint_liks_bpf, joint_true_logliks)))
        joint_logliks_deviations_apf.append(np.average(mse(joint_liks_apf, joint_true_logliks)))
        joint_logliks_deviations_iapf.append(np.average(mse(joint_liks_iapf, joint_true_logliks)))
        joint_logliks_deviations_oapf.append(np.average(mse(joint_liks_oapf, joint_true_logliks)))
        joint_logliks_deviations_faapf.append(np.average(mse(joint_liks_faapf, joint_true_logliks)))

    all_mean_deviations_bpf.append(mean_deviations_bpf)
    all_mean_deviations_apf.append(mean_deviations_apf)
    all_mean_deviations_iapf.append(mean_deviations_iapf)
    all_mean_deviations_oapf.append(mean_deviations_oapf)
    all_mean_deviations_faapf.append(mean_deviations_faapf)
    all_mean_deviations_approx_kf.append(mean_deviations_approx_kf)

    all_joint_logliks_deviations_bpf.append(joint_logliks_deviations_bpf)
    all_joint_logliks_deviations_apf.append(joint_logliks_deviations_apf)
    all_joint_logliks_deviations_iapf.append(joint_logliks_deviations_iapf)
    all_joint_logliks_deviations_oapf.append(joint_logliks_deviations_oapf)
    all_joint_logliks_deviations_faapf.append(joint_logliks_deviations_faapf)



res_means = np.vstack([
    all_mean_deviations_bpf,
    all_mean_deviations_apf,
    all_mean_deviations_iapf,
    all_mean_deviations_oapf,
    all_mean_deviations_faapf,
    all_mean_deviations_approx_kf
])

res_joint_logliks = np.vstack([
    all_joint_logliks_deviations_bpf,
    all_joint_logliks_deviations_apf,
    all_joint_logliks_deviations_iapf,
    all_joint_logliks_deviations_oapf,
    all_joint_logliks_deviations_faapf
])

# REDUCED
np.savetxt('results/lingauss/joint_logliks/moreparticles_results_lingauss_withoptimal_reduced5_particles-dim'+str(dim)+'-trvar'+str(trans_var)+'-obsvar'+str(obs_var)+'.out', res_joint_logliks, delimiter=',')
np.savetxt('results/lingauss/means/moreparticles_results_lingauss_withoptimal_reduced5_particles-dim'+str(dim)+'-trvar'+str(trans_var)+'-obsvar'+str(obs_var)+'.out', res_means, delimiter=',')

