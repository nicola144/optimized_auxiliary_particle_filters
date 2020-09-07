import sys
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.optimize import nnls
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
from utils import *

# seeds
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Decides whether to perform reduction of the optimization problem given by OAPF
reduce=True

#     ParticleFilter
#       /       \
# BPF/APF etc   Lin/Gauss/StochVol etc
#       \        /
#      Concrete PF
#

# NOTE : Originally inspired by implementation in : https://github.com/ctgk/PRML/blob/master/prml/markov/particle.py
class ParticleFilter(ABC):

    def __init__(self, init_particle):

        self.particle = [init_particle]
        self.n_particle, self.ndim_hidden = init_particle.shape
        self.importance_weight = [np.log(np.ones(self.n_particle) / self.n_particle)]
        self.indices = []

    @abstractmethod
    def propagate(self, particles):
        pass

    @abstractmethod
    def transition_density(self, at, mean, **params):
        pass

    @abstractmethod
    def observation_density(self, obs, mean, **params):
        pass

    @abstractmethod
    def importance_weight_function(self, observed):
        pass

    @abstractmethod
    def simulation_weight_function(self, observed):
        pass
    # Of course, one could use more advanced resampling schemes
    def multinomial_resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=normalize_log(self.simulation_weight[-1]))
        self.indices.append(index)
        return self.particle[-1][index], index

    # Bring particles forward
    def simulate(self, observed):
        self.simulation_weight_function(observed)
        resampled, indices = self.multinomial_resample()
        propagated = self.propagate(resampled)
        self.particle.append(propagated)
        return indices

    # Weight particles
    def weight(self, observed):
        self.importance_weight_function(observed)
        return self.particle[-1], self.importance_weight[-1]

    def filter(self, observed_sequence):
        mean = []
        cov = []
        logz_estimates = []
        ess = []
        n_unique_sequence = []
        w_vars = []

        for obs in tqdm(observed_sequence):
            # Can be used for unique particle count
            indices = self.simulate(obs)
            p, w = self.weight(obs)
            normalized_w = normalize_log(w)
            # ESS estimate uses normalized weights
            ess_est = 1. / normalized_w.T.dot(normalized_w)
            logz_estimate = logsumexp(np.log(1. / self.n_particle) + w) #not sure this works
            n_unique = np.unique(indices).shape[0]
            w_var = np.var(w)

            if np.isnan(normalized_w).any():
                print('some weights are nan')
                sys.exit()

            mean.append(np.average(p, axis=0, weights=normalized_w))
            cov.append(np.cov(p, rowvar=False, aweights=normalized_w))
            ess.append(ess_est)
            logz_estimates.append(logz_estimate)
            n_unique_sequence.append(n_unique)
            w_vars.append(w_var)

        return np.asarray(mean), np.asarray(cov), np.asarray(logz_estimates), np.asarray(ess), np.asarray(
            n_unique_sequence), np.array(w_vars)


class BPF(ParticleFilter):
    def __init__(self, **params):
        super().__init__(**params)
        self.simulation_weight = []
        self.prev_centers_list = []

    def importance_weight_function(self, observed):
        unnormalized = self.observation_density(obs=observed, mean=self.particle[-1], offset=self.observation_offset)
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self, observed):
        self.simulation_weight.append(self.importance_weight[-1])


class APF(ParticleFilter):
    def __init__(self, **params):
        super().__init__(**params)
        self.simulation_weight = []
        self.prev_centers_list = []

    def importance_weight_function(self, observed):
        prev_centers = self.prev_centers_list[-1]
        unnormalized = self.observation_density(obs=observed, mean=self.particle[-1],
                                                offset=self.observation_offset) - self.observation_density(obs=observed,
                                                                                                           mean=prev_centers,
                                                                                                           offset=self.observation_offset)
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self, observed):
        prev_centers = self.compute_prev_centers()
        unnormalized = self.importance_weight[-1] + self.observation_density(obs=observed, mean=prev_centers,
                                                                             offset=self.observation_offset)
        self.simulation_weight.append(unnormalized)


class IAPF(ParticleFilter):

    def __init__(self, **params):
        super().__init__(**params)
        self.simulation_weight = []
        self.prev_centers_list = []

    # This is really the same for IAPF and OAPF
    def importance_weight_function(self, observed):
        prev_centers = self.prev_centers_list[-1]
        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers)
        predictive = logmatmulexp(kernels, np.array(self.importance_weight[-1]))
        proposal = logmatmulexp(kernels, np.array(self.simulation_weight[-1]))
        lik = self.observation_density(obs=observed, mean=self.particle[-1], offset=self.observation_offset)
        unnormalized = lik + predictive - proposal
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self, observed):
        prev_centers = self.compute_prev_centers()
        kernels_at_centers = self.transition_density(at=prev_centers, mean=prev_centers)
        pred_lik = self.observation_density(obs=observed, mean=prev_centers, offset=self.observation_offset)
        sum_numerator = logmatmulexp(kernels_at_centers, np.array(self.importance_weight[-1]))
        sum_denominator = logmatmulexp(kernels_at_centers, np.ones(kernels_at_centers.shape[0]))
        unnormalized = pred_lik + sum_numerator - sum_denominator
        self.simulation_weight.append(unnormalized)


class OAPF(ParticleFilter):

    def __init__(self, **params):
        super().__init__(**params)
        self.simulation_weight = []
        self.prev_centers_list = []

    # This is really the same for IAPF and OAPF
    def importance_weight_function(self, observed):
        prev_centers = self.prev_centers_list[-1]

        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers)

        predictive = logmatmulexp(kernels, np.array(self.importance_weight[-1]))
        proposal = logmatmulexp(kernels, np.array(self.simulation_weight[-1]))

        lik = self.observation_density(obs=observed, mean=self.particle[-1], offset=self.observation_offset)

        unnormalized = lik + predictive - proposal
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self, observed):
        prev_centers = self.compute_prev_centers()

        kernels_at_centers = self.transition_density(at=prev_centers, mean=prev_centers)

        pred_lik = self.observation_density(obs=observed, mean=prev_centers, offset=self.observation_offset)

        scaled_kernels = pred_lik.reshape(-1, 1) + kernels_at_centers

        logA = kernels_at_centers

        logb = logmatmulexp(scaled_kernels, np.array(self.importance_weight[-1]))

        A = np.exp(logA) + 0.1 * np.eye(logb.shape[0])
        b = np.exp(logb)

        if reduce:
            unnormalized = np.zeros(b.shape)
            smaller_A,smaller_b, indices_tokeep = reduce_system(self.n_particle,A,b)
            res = nnls(smaller_A,smaller_b)[0]
            np.add.at(unnormalized, indices_tokeep, res)
        else:
            unnormalized = nnls(A, b)[0]

        sanity_checks(unnormalized)

        # will trigger warning about taking log of 0. it's fine
        # since subsequent functions can handle -np.inf
        to_append = np.log(unnormalized)
        self.simulation_weight.append(to_append)


class LinearGaussianPF(ParticleFilter):
    def __init__(self, init_particle,  transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super().__init__(init_particle)

        self.transition_cov = transition_cov
        self.observation_cov = observation_cov
        self.transition_offset = transition_offset
        self.observation_offset = observation_offset
        self.transition_mat = transition_mat
        self.observation_mat = observation_mat

    def compute_prev_centers(self):
        return self.particle[-1].dot(self.transition_mat) + self.transition_offset

    def transition_density(self, at, mean, **params):
        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(self.transition_cov)
        at = np.expand_dims(at, axis=1)
        at = torch.from_numpy(at)
        cov_all = cov[None, ...].repeat_interleave(self.n_particle, 0)

        kernels = MultivariateNormal(mean, cov_all).log_prob(at)

        return kernels.numpy()

    def observation_density(self, obs, mean, **params):
        mean_all = np.matmul(np.array(mean), self.observation_mat) + params['offset']
        obs = torch.from_numpy(obs)
        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)
        mean_all = torch.from_numpy(mean_all)
        obs_cov = torch.from_numpy(self.observation_cov)

        liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()

    def propagate(self, resmpled_particles):
        prev_centers = resmpled_particles.dot(self.transition_mat) + self.transition_offset

        self.prev_centers_list.append(prev_centers)
        res = prev_centers + np.random.multivariate_normal(mean=np.zeros(self.ndim_hidden), cov=self.transition_cov,
                                                           size=self.n_particle)
        return res


class LinearGaussianBPF(BPF, LinearGaussianPF):
    def __init__(self, init_particle, transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super(LinearGaussianBPF, self).__init__(init_particle=init_particle,
                                                transition_cov=transition_cov,
                                                observation_cov=observation_cov,
                                                transition_mat=transition_mat,
                                                transition_offset=transition_offset,
                                                observation_mat=observation_mat,
                                                observation_offset=observation_offset)


class LinearGaussianAPF(APF, LinearGaussianPF):
    def __init__(self, init_particle,  transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super(LinearGaussianAPF, self).__init__(init_particle=init_particle,
                                                transition_cov=transition_cov,
                                                observation_cov=observation_cov,
                                                transition_mat=transition_mat,
                                                transition_offset=transition_offset,
                                                observation_mat=observation_mat,
                                                observation_offset=observation_offset)


class LinearGaussianIAPF(IAPF, LinearGaussianPF):
    def __init__(self, init_particle,  transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super(LinearGaussianIAPF, self).__init__(init_particle=init_particle,
                                                 transition_cov=transition_cov,
                                                 observation_cov=observation_cov,
                                                 transition_mat=transition_mat,
                                                 transition_offset=transition_offset,
                                                 observation_mat=observation_mat,
                                                 observation_offset=observation_offset)


class LinearGaussianOAPF(OAPF, LinearGaussianPF):
    def __init__(self, init_particle, transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super(LinearGaussianOAPF, self).__init__(init_particle=init_particle,
                                                   transition_cov=transition_cov,
                                                   observation_cov=observation_cov,
                                                   transition_mat=transition_mat,
                                                   transition_offset=transition_offset,
                                                   observation_mat=observation_mat,
                                                   observation_offset=observation_offset)


class StochVolPF(ParticleFilter):
    def __init__(self, init_particle, transition_cov, transition_offset, phi):
        super().__init__(init_particle)
        self.transition_cov = transition_cov
        self.transition_offset = transition_offset
        self.phi = phi
        self.observation_offset = np.zeros(self.ndim_hidden,)

    def compute_prev_centers(self):
        return np.matmul(self.particle[-1] - self.transition_offset, self.phi) + self.transition_offset

    def transition_density(self, at, mean, **params):
        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(self.transition_cov)
        at = np.expand_dims(at, axis=1)
        at = torch.from_numpy(at)
        cov_all = cov[None, ...].repeat_interleave(self.n_particle, 0)

        kernels = MultivariateNormal(mean, cov_all).log_prob(at)

        return kernels.numpy()

    def observation_density(self, obs, mean, **params):

        # this is specific to our stochastic vol. model
        actual_mean = np.zeros((self.n_particle, self.ndim_hidden))
        obs = torch.from_numpy(obs).double()
        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)
        actual_mean_all = torch.from_numpy(actual_mean).double()
        obs_cov = np.exp(mean)
        obs_cov = torch.from_numpy(np.apply_along_axis(np.diag, 1, obs_cov)).double()

        liks = MultivariateNormal(actual_mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()

    def propagate(self, resampled_particles):
        prev_centers = np.matmul(resampled_particles - self.transition_offset, self.phi) + self.transition_offset

        self.prev_centers_list.append(prev_centers)
        res = prev_centers + np.random.multivariate_normal(mean=np.zeros(self.ndim_hidden), cov=self.transition_cov,
                                                           size=self.n_particle)
        return res


class StochVolBPF(BPF, StochVolPF):
    def __init__(self, init_particle, transition_cov, transition_offset, phi):
        super(StochVolBPF, self).__init__(init_particle=init_particle,
                                          transition_cov=transition_cov,
                                          transition_offset=transition_offset,
                                          phi=phi)


class StochVolAPF(APF, StochVolPF):
    def __init__(self, init_particle, transition_cov, transition_offset, phi):
        super(StochVolAPF, self).__init__(init_particle=init_particle,
                                          transition_cov=transition_cov,
                                          transition_offset=transition_offset,
                                          phi=phi)


class StochVolIAPF(IAPF, StochVolPF):
    def __init__(self, init_particle, transition_cov, transition_offset, phi):
        super(StochVolIAPF, self).__init__(init_particle=init_particle,
                                           transition_cov=transition_cov,
                                           transition_offset=transition_offset,
                                           phi=phi)


class StochVolOAPF(OAPF, StochVolPF):
    def __init__(self, init_particle, transition_cov, transition_offset, phi):
        super(StochVolOAPF, self).__init__(init_particle=init_particle,
                                             transition_cov=transition_cov,
                                             transition_offset=transition_offset,
                                             phi=phi)
