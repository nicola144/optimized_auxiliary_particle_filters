import sys
from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.optimize import nnls,linprog,least_squares
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import *
import time
# pytorch seed
torch.manual_seed(0)

# Decides whether to perform reduction of the optimization problem given by OAPF
reduce=True

# Class structure

#     ParticleFilter
#       /       \
# BPF/APF etc   Lin-Gauss/StochVol or others
#       \        /
#      Concrete PF !
#

# NOTE : Originally inspired by implementation in : https://github.com/ctgk/PRML/blob/master/prml/markov/particle.py
class ParticleFilter(ABC):

    def __init__(self, init_particle, random_state):
        """
        Initializer for generic particle filter. 
        
        :param init_particle: initialized particles drawn from a prior distribution 
        :type init_particle: A list or np.array of shape N x D , N being n. of particles, D the dimension of latent state
        :param random_state: A np.random object to hold the random seed 
        :type random_state: 
        """

        self.particle = [init_particle]
        self.n_particle, self.ndim_hidden = init_particle.shape
        self.importance_weight = [np.log(np.ones(self.n_particle) / self.n_particle)]
        self.indices = []
        self.random_state = random_state


    @abstractmethod
    def propagate(self, particles):
        pass

    @abstractmethod
    def transition_density(self, at, mean, **params):
        pass

    @abstractmethod
    def observation_density(self, obs, **params):
        pass

    @abstractmethod
    def importance_weight_function(self, obs):
        pass

    @abstractmethod
    def simulation_weight_function(self, obs):
        pass


    def multinomial_resample(self):
        """

        :param self:
        :type self:
        :return:
        :rtype:
        """
        # print('resampling')
        index = self.random_state.choice(self.n_particle, self.n_particle, p=normalize_log(self.simulation_weight[-1]))
        self.indices.append(index)
        return self.particle[-1][index], index

    # Bring particles forward
    def simulate(self, obs):
        """

        :param self:
        :type self:
        :param obs:
        :type obs:
        :return:
        :rtype:
        """
        self.simulation_weight_function(obs)
        resampled, indices = self.multinomial_resample()

        if self.is_optimal:
            propagated = self.propagate_optimal(resampled,obs)
        else:
            propagated = self.propagate(resampled)
        # print('appending new particles')
        self.particle.append(propagated)
        return indices

    # Weight particles
    def weight(self, obs):
        """

        :param self:
        :type self:
        :param obs:
        :type obs:
        :return:
        :rtype:
        """
        self.importance_weight_function(obs)
        return self.particle[-1], self.importance_weight[-1]

    def filter(self, observed_sequence):
        """

        :param self:
        :type self:
        :param observed_sequence:
        :type observed_sequence:
        :return:
        :rtype:
        """
        mean = []
        cov = []
        logz_estimates = []
        joint_logz_estimates = []
        ess = []
        n_unique_sequence = []
        w_vars = []
        times = []

        for obs in observed_sequence:
            # Can be used for unique particle count
            start = time.time()
            indices = self.simulate(obs)
            p, w = self.weight(obs)
            end = time.time()
            times.append(end-start)
            normalized_w = normalize_log(w)
            # ESS estimate uses normalized weights
            ess_est = get_ess(log_normalize_log(w))
            # Estimate of log normalizing constant
            logz = self.compute_logz(w, self.simulation_weight[-1])
            # Estimate of joint logz
            joint_logz = logz + np.sum(logz_estimates)
            # Unique particles
            n_unique = 0
            # n_unique = np.unique(indices).shape[0]
            # Sample variance. Bessel's correction
            w_var = np.var(w,ddof=1)

            if np.isnan(normalized_w).any():
                print('some weights are nan')
                sys.exit()

            mean.append(np.average(p, axis=0, weights=normalized_w))
            cov.append(np.cov(p, rowvar=False, aweights=normalized_w))
            ess.append(ess_est)
            n_unique_sequence.append(n_unique)
            w_vars.append(w_var)
            logz_estimates.append(logz)
            joint_logz_estimates.append(joint_logz)

        return np.asarray(mean), np.asarray(cov),  np.asarray(ess), np.asarray(
            n_unique_sequence), np.asarray(w_vars), np.asarray(logz_estimates), np.asarray(joint_logz_estimates), np.asarray(times)

class BPF(ParticleFilter):
    def __init__(self, **params):
        super().__init__(**params)
        self.is_optimal = False
        self.simulation_weight = []
        # not needed , a design hack
        # ----------------------------------------------------
        self.prev_centers_list_resampled = []
        self.prev_centers_list_non_resampled = []
        # ----------------------------------------------------

    def importance_weight_function(self, observed):
        unnormalized = self.observation_density(obs=observed, mean=self.particle[-1], offset=self.observation_offset)
        # self.importance_weight[-1] = unnormalized
        self.importance_weight.append(unnormalized)

    def simulation_weight_function(self, observed):
        # not needed , a design hack
        # ----------------------------------------------------
        prev_centers = self.compute_prev_centers(self.particle[-1])
        self.prev_centers_list_non_resampled.append(prev_centers)
        # ----------------------------------------------------

        self.simulation_weight.append(self.importance_weight[-1])

    def compute_logz(self,w,l):
        #not using l for bpf
        return logsumexp(w) - np.log(self.n_particle)

class APF(ParticleFilter):
    def __init__(self, **params):
        super().__init__(**params)
        self.is_optimal = False
        self.simulation_weight = []
        self.prev_centers_list_resampled = []
        self.prev_centers_list_non_resampled = []
        self.pred_liks = []

        self.pred_liks_resampled = []

    def importance_weight_function(self, observed):
        prev_centers = self.prev_centers_list_resampled[-1]
        pred_lik_resampled = self.observation_density(obs=observed, mean=prev_centers,
                                            offset=self.observation_offset)
        self.pred_liks_resampled.append(pred_lik_resampled)

        unnormalized = self.observation_density(obs=observed, mean=self.particle[-1],
                                                offset=self.observation_offset) - self.observation_density(obs=observed,
                                                                                                           mean=prev_centers,
                                                                                                           offset=self.observation_offset)
        self.importance_weight.append(unnormalized)

    def simulation_weight_function(self, observed):
        # print('simulation')
        # print('computing centers')
        prev_centers = self.compute_prev_centers(self.particle[-1])
        self.prev_centers_list_non_resampled.append(prev_centers)

        pred_lik = self.observation_density(obs=observed, mean=prev_centers,
                                            offset=self.observation_offset)
        self.pred_liks.append(pred_lik)
        unnormalized = self.importance_weight[-1] + pred_lik
        self.simulation_weight.append(unnormalized)

    def compute_logz(self,w,l):
        # this is correct for apf
        return logsumexp(w) + logsumexp( log_normalize_log(self.importance_weight[-2]) + self.pred_liks[-1]) - np.log(self.n_particle)


class FullyAdaptedAPF(ParticleFilter):
    def __init__(self, **params):
        super().__init__(**params)
        self.simulation_weight = []
        self.prev_centers_list_resampled = []
        self.prev_centers_list_non_resampled = []
        self.pred_liks = []

        self.is_optimal = True

        self.sigma_y_inv = safe_invert(self.observation_cov)
        self.sigma_x_inv = safe_invert(self.transition_cov)
        optimal_kernel_cov_inv = self.sigma_x_inv + self.observation_mat.T.dot( self.sigma_y_inv.dot(self.observation_mat)  )
        self.optimal_kernel_cov = safe_invert(optimal_kernel_cov_inv)

        self.pred_lik_cov = self.transition_cov + self.observation_mat.dot(self.observation_cov.dot( self.observation_mat.T ))

    def importance_weight_function(self, observed):

        unnormalized = np.zeros(self.n_particle)
        self.importance_weight.append(unnormalized)

    def simulation_weight_function(self, observed):

        prev_centers = self.compute_prev_centers(self.particle[-1])
        self.prev_centers_list_non_resampled.append(prev_centers)

        # mean_pred_lik = self.observation_mat.dot(prev_centers.T).T
        mean_pred_lik = prev_centers.dot(self.observation_mat)
        pred_lik = self.optimal_lambda_density(obs=observed, mean=mean_pred_lik)

        self.pred_liks.append(pred_lik)

        unnormalized = pred_lik - np.log(self.n_particle)
        self.simulation_weight.append(unnormalized)

    def compute_logz(self,w,l):
        # this is correct for faapf
        
        return logsumexp(  self.pred_liks[-1] ) - np.log(self.n_particle)
        # return logsumexp(w) - np.log(self.n_particle)
        # return logsumexp(w) + logsumexp( log_normalize_log(self.importance_weight[-2]) + self.pred_liks[-1]) - np.log(self.n_particle)

    def optimal_lambda_density(self,obs,mean):

        obs = torch.from_numpy(obs).double()
        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)
        mean_all = torch.from_numpy(mean).double()
        obs_cov = torch.from_numpy(self.pred_lik_cov).double()
        liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()



class IAPF(ParticleFilter):

    def __init__(self, **params):
        super().__init__(**params)
        self.is_optimal = False
        self.simulation_weight = []
        self.prev_centers_list_resampled = []
        self.prev_centers_list_non_resampled = []

    # This is really the same for IAPF and OAPF
    def importance_weight_function(self, observed):
        # here, need to use centers of non-resampled particles
        prev_centers = self.prev_centers_list_non_resampled[-1]
        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers)
        predictive = logmatmulexp(kernels, np.array(log_normalize_log(self.importance_weight[-1])))
        proposal = logmatmulexp(kernels, np.array(self.simulation_weight[-1]))
        lik = self.observation_density(obs=observed, mean=self.particle[-1], offset=self.observation_offset)
        unnormalized = lik + predictive - proposal

        self.importance_weight.append(unnormalized)

    def simulation_weight_function(self, observed):
        # this computes centers for non resampled particles
        prev_centers = self.compute_prev_centers(self.particle[-1])
        self.prev_centers_list_non_resampled.append(prev_centers)
        kernels_at_centers = self.transition_density(at=prev_centers, mean=prev_centers)
        pred_lik = self.observation_density(obs=observed, mean=prev_centers, offset=self.observation_offset)
        sum_numerator = logmatmulexp(kernels_at_centers, np.array(log_normalize_log(self.importance_weight[-1])))
        sum_denominator = logmatmulexp(kernels_at_centers, np.ones(kernels_at_centers.shape[0]))
        unnormalized = pred_lik + sum_numerator - sum_denominator
        self.simulation_weight.append(unnormalized)

    def compute_logz(self,w,l):
        return logsumexp(w) + logsumexp( l ) - np.log(self.n_particle)

class OAPF(ParticleFilter):

    def __init__(self, **params):
        super().__init__(**params)
        self.is_optimal = False
        self.simulation_weight = []
        self.prev_centers_list_resampled = []
        self.prev_centers_list_non_resampled = []

    # This is really the same for IAPF and OAPF
    def importance_weight_function(self, observed):
        prev_centers = self.prev_centers_list_non_resampled[-1]

        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers)

        predictive = logmatmulexp(kernels, np.array(log_normalize_log(self.importance_weight[-1])))
        proposal = logmatmulexp(kernels, np.array(self.simulation_weight[-1]))

        lik = self.observation_density(obs=observed, mean=self.particle[-1], offset=self.observation_offset)

        unnormalized = lik + predictive - proposal

        self.importance_weight.append(unnormalized)

    def simulation_weight_function(self, observed):
        prev_centers = self.compute_prev_centers(self.particle[-1])
        self.prev_centers_list_non_resampled.append(prev_centers)

        kernels_at_centers = self.transition_density(at=prev_centers, mean=prev_centers)

        pred_lik = self.observation_density(obs=observed, mean=prev_centers, offset=self.observation_offset)

        scaled_kernels = pred_lik.reshape(-1, 1) + kernels_at_centers

        logA = kernels_at_centers

        logb = logmatmulexp(scaled_kernels, np.array(log_normalize_log(self.importance_weight[-1])))

        # Conditioning ?
        A = np.exp(logA) + np.eye(logb.shape[0]) * 1e-9
        b = np.exp(logb)

        # if not check_symmetric(A):
        #     print('not symm')
        #     sys.exit()


        if reduce:
            unnormalized = np.zeros(b.shape)
            smaller_A,smaller_b, indices_tokeep = reduce_system(self.n_particle,A,b)
            res = nnls(smaller_A,smaller_b)[0]
            np.add.at(unnormalized, indices_tokeep, res)
        else:
            unnormalized = nnls(A, b)[0]
            # Or can use simplex/ interior point
            # A = np.hstack((A, -np.eye(b.shape[0])))
            # c = np.concatenate(( np.zeros(b.shape[0]), np.ones(b.shape[0])  ))
            # results = linprog(c=c, A_eq=A, b_eq=b, bounds=[(0,None)]*b.shape[0]*2, method='interior-point',options={'presolve':True, 'sparse':True}) # ,options={'presolve':False} can be interior-point or revised simplex
            # result_vec = results['x']
            # unnormalized = result_vec[:b.shape[0]]


        # unnormalized = randomized_nnls(A, b, self.n_particle)

        sanity_checks(unnormalized)

        # will trigger warning about taking log of 0. it's fine
        # since subsequent functions can handle -np.inf
        to_append = np.log(unnormalized)
        self.simulation_weight.append(to_append)

    def compute_logz(self,w,l):
        return logsumexp(w) + logsumexp( l ) - np.log(self.n_particle)


class LinearGaussianPF(ParticleFilter):
    def __init__(self, init_particle, random_state, transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super().__init__(init_particle, random_state)

        self.transition_cov = transition_cov
        self.observation_cov = observation_cov
        self.transition_offset = transition_offset
        self.observation_offset = observation_offset
        self.transition_mat = transition_mat
        self.observation_mat = observation_mat

    def compute_prev_centers(self, particle):
        return particle.dot(self.transition_mat) + self.transition_offset

    def transition_density(self, at, mean, **params):
        mean = torch.from_numpy(mean).double()
        cov = torch.from_numpy(self.transition_cov).double()
        at = np.expand_dims(at, axis=1)
        at = torch.from_numpy(at).double()
        cov_all = cov[None, ...].repeat_interleave(self.n_particle, 0)

        kernels = MultivariateNormal(mean, cov_all).log_prob(at)

        return kernels.numpy()

    def observation_density(self, obs, **params):
        mean = params['mean']
        mean_all = np.matmul(np.array(mean), self.observation_mat) + params['offset']
        obs = torch.from_numpy(obs).double()
        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)
        mean_all = torch.from_numpy(mean_all).double()
        obs_cov = torch.from_numpy(self.observation_cov).double()
        liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()

    def propagate(self, resampled_particles):
        # print('propagation')
        # print('using centers to propagate')
        # this computes centers for resampled particles
        prev_centers = self.compute_prev_centers(resampled_particles)
        self.prev_centers_list_resampled.append(prev_centers)

        res = prev_centers + self.random_state.multivariate_normal(mean=np.zeros(self.ndim_hidden), cov=self.transition_cov,
                                                           size=self.n_particle)

        return res

    def propagate_optimal(self, resampled_particles, obs):

        obs = np.tile(obs,(self.n_particle,1))

        prev_centers = self.compute_prev_centers(resampled_particles)
        self.prev_centers_list_resampled.append(prev_centers)

        right = obs.dot(self.sigma_y_inv).dot(self.observation_mat)

        temp = prev_centers.dot(self.sigma_x_inv) + right

        mean_optimal = temp.dot(self.optimal_kernel_cov)

        res = mean_optimal + self.random_state.multivariate_normal(mean=np.zeros(self.ndim_hidden), cov=self.optimal_kernel_cov,
                                                                   size=self.n_particle)
        return res

class LinearGaussianFullyAdapted(FullyAdaptedAPF, LinearGaussianPF):
    def __init__(self, init_particle, random_state, transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super(LinearGaussianFullyAdapted, self).__init__(
                                                init_particle=init_particle,
                                                random_state=random_state,
                                                transition_cov=transition_cov,
                                                observation_cov=observation_cov,
                                                transition_mat=transition_mat,
                                                transition_offset=transition_offset,
                                                observation_mat=observation_mat,
                                                observation_offset=observation_offset)

class LinearGaussianBPF(BPF, LinearGaussianPF):
    def __init__(self, init_particle, random_state, transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super(LinearGaussianBPF, self).__init__(init_particle=init_particle,
                                                random_state=random_state,
                                                transition_cov=transition_cov,
                                                observation_cov=observation_cov,
                                                transition_mat=transition_mat,
                                                transition_offset=transition_offset,
                                                observation_mat=observation_mat,
                                                observation_offset=observation_offset)

class LinearGaussianAPF(APF, LinearGaussianPF):
    def __init__(self, init_particle, random_state, transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super(LinearGaussianAPF, self).__init__(init_particle=init_particle,
                                                random_state=random_state,
                                                transition_cov=transition_cov,
                                                observation_cov=observation_cov,
                                                transition_mat=transition_mat,
                                                transition_offset=transition_offset,
                                                observation_mat=observation_mat,
                                                observation_offset=observation_offset)

class LinearGaussianIAPF(IAPF, LinearGaussianPF):
    def __init__(self, init_particle, random_state, transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super(LinearGaussianIAPF, self).__init__(init_particle=init_particle,
                                                 random_state=random_state,
                                                 transition_cov=transition_cov,
                                                 observation_cov=observation_cov,
                                                 transition_mat=transition_mat,
                                                 transition_offset=transition_offset,
                                                 observation_mat=observation_mat,
                                                 observation_offset=observation_offset)

class LinearGaussianOAPF(OAPF, LinearGaussianPF):
    def __init__(self, init_particle, random_state, transition_cov, observation_cov, transition_mat, transition_offset,
                 observation_mat, observation_offset):
        super(LinearGaussianOAPF, self).__init__(init_particle=init_particle,
                                                 random_state=random_state,
                                                 transition_cov=transition_cov,
                                                   observation_cov=observation_cov,
                                                   transition_mat=transition_mat,
                                                   transition_offset=transition_offset,
                                                   observation_mat=observation_mat,
                                                   observation_offset=observation_offset)


class StochVolPF(ParticleFilter):
    def __init__(self, init_particle, random_state, transition_cov, transition_offset, phi):
        super().__init__(init_particle, random_state)
        self.transition_cov = transition_cov
        self.transition_offset = transition_offset
        self.phi = phi
        self.observation_offset = np.zeros(self.ndim_hidden,)

    def compute_prev_centers(self, particle):
        return np.matmul(particle - self.transition_offset, self.phi) + self.transition_offset

    def transition_density(self, at, mean, **params):
        mean = torch.from_numpy(mean).double()
        cov = torch.from_numpy(self.transition_cov).double()
        at = np.expand_dims(at, axis=1)
        at = torch.from_numpy(at).double()
        cov_all = cov[None, ...].repeat_interleave(self.n_particle, 0)

        kernels = MultivariateNormal(mean, cov_all).log_prob(at)

        return kernels.numpy()

    def observation_density(self, obs, **params):
        mean = params['mean']
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
        prev_centers = self.compute_prev_centers(resampled_particles)
        self.prev_centers_list_resampled.append(prev_centers)

        res = prev_centers + self.random_state.multivariate_normal(mean=np.zeros(self.ndim_hidden), cov=self.transition_cov,
                                                           size=self.n_particle)
        return res


class StochVolBPF(BPF, StochVolPF):
    def __init__(self, init_particle, random_state, transition_cov, transition_offset, phi):
        super(StochVolBPF, self).__init__(init_particle=init_particle,
                                          random_state=random_state,
                                          transition_cov=transition_cov,
                                          transition_offset=transition_offset,
                                          phi=phi)

class StochVolAPF(APF, StochVolPF):
    def __init__(self, init_particle, random_state, transition_cov, transition_offset, phi):
        super(StochVolAPF, self).__init__(init_particle=init_particle,
                                          random_state=random_state,
                                          transition_cov=transition_cov,
                                          transition_offset=transition_offset,
                                          phi=phi)

class StochVolIAPF(IAPF, StochVolPF):
    def __init__(self, init_particle, random_state, transition_cov, transition_offset, phi):
        super(StochVolIAPF, self).__init__(init_particle=init_particle,
                                           random_state=random_state,
                                           transition_cov=transition_cov,
                                           transition_offset=transition_offset,
                                           phi=phi)

class StochVolOAPF(OAPF, StochVolPF):
    def __init__(self, init_particle, random_state, transition_cov, transition_offset, phi):
        super(StochVolOAPF, self).__init__(init_particle=init_particle,
                                           random_state=random_state,
                                           transition_cov=transition_cov,
                                             transition_offset=transition_offset,
                                             phi=phi)


class LorenzPF(ParticleFilter):
    def __init__(self, init_particle, random_state, s, r, b, delta, transition_cov, observation_var):
        super().__init__(init_particle, random_state)
        self.s = s
        self.r = r
        self.b = b
        self.delta = delta
        self.transition_cov = transition_cov
        self.observation_var = observation_var
        self.observation_offset = None

    def compute_prev_centers(self, particle):
        x = particle[:, 0]
        y = particle[:, 1]
        z = particle[:, 2]

        x_dot = self.s * ( y - x )
        y_dot = self.r * x - y - x * z
        z_dot = x * y - self.b * z

        xs = x + (x_dot * self.delta)
        ys = y + (y_dot * self.delta)
        zs = z + (z_dot * self.delta)

        # stack dimensions together
        res = np.vstack([xs, ys, zs]).T
        return res

    def transition_density(self, at, mean, **params):
        mean = torch.from_numpy(mean).double()
        cov = torch.from_numpy(self.transition_cov).double()
        at = np.expand_dims(at, axis=1)
        at = torch.from_numpy(at).double()
        cov_all = cov[None, ...].repeat_interleave(self.n_particle, 0)

        kernels = MultivariateNormal(mean, cov_all).log_prob(at)

        return kernels.numpy()

    def observation_density(self, obs, **params):
        mean = params['mean']
        # The Lorenz system is observed through its first coordinate only
        mean = mean[:,0].reshape(-1,1)
        obs = np.atleast_1d(obs)
        obs = torch.from_numpy(obs).double()
        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)
        mean_all = torch.from_numpy(mean).double()
        obs_cov = torch.from_numpy(self.observation_var).double()
        liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()

    def propagate(self, resampled_particles):
        prev_centers = self.compute_prev_centers(resampled_particles)
        self.prev_centers_list_resampled.append(prev_centers)

        res = prev_centers + self.random_state.multivariate_normal(mean=np.zeros(self.ndim_hidden), cov=self.transition_cov,
                                                                   size=self.n_particle)
        return res


class LorenzBPF(BPF, LorenzPF):
    def __init__(self, init_particle, random_state, s, r, b, delta, transition_cov, observation_var):
        super(LorenzBPF, self).__init__(init_particle=init_particle,
                                        random_state=random_state,
                                        s=s,
                                        r=r,
                                        b=b,
                                        delta=delta,
                                        transition_cov=transition_cov,
                                        observation_var=observation_var)

class LorenzAPF(APF, LorenzPF):
    def __init__(self, init_particle, random_state, s, r, b, delta, transition_cov, observation_var):
        super(LorenzAPF, self).__init__(init_particle=init_particle,
                                        random_state=random_state,
                                        s=s,
                                        r=r,
                                        b=b,
                                        delta=delta,
                                        transition_cov=transition_cov,
                                        observation_var=observation_var)

class LorenzIAPF(IAPF, LorenzPF):
    def __init__(self, init_particle, random_state, s, r, b, delta, transition_cov, observation_var):
        super(LorenzIAPF, self).__init__(init_particle=init_particle,
                                        random_state=random_state,
                                        s=s,
                                        r=r,
                                        b=b,
                                        delta=delta,
                                        transition_cov=transition_cov,
                                        observation_var=observation_var)

class LorenzOAPF(OAPF, LorenzPF):
    def __init__(self, init_particle, random_state, s, r, b, delta, transition_cov, observation_var):
        super(LorenzOAPF, self).__init__(init_particle=init_particle,
                                        random_state=random_state,
                                        s=s,
                                        r=r,
                                        b=b,
                                        delta=delta,
                                        transition_cov=transition_cov,
                                        observation_var=observation_var)