import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal as multivariate_normal_scipy
from scipy.stats import lognorm as lognorm
from sklearn.datasets import make_spd_matrix
import sys
from tqdm import tqdm
from scipy.optimize import nnls,linprog,minimize
from scipy.optimize import least_squares
# from scipy.linalg import svd
import scipy
from numpy.linalg import cond
from utils import *
import torch 
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal
import re 
import time
import warnings

from numpy import ma

torch.manual_seed(random_seed)
np.random.seed(random_seed)



# Abstract base class for a particle filter 
class ParticleFilter(ABC):

    def __init__(self, init_particle, random_state):
        """
        Parameters
        ----------
        init_particle: n_particle, ndim_hidden) np.ndarray
        			   initial hidden state
       	random_state: 
		"""
        self.particle = [init_particle]
        self.n_particle, self.ndim_hidden = init_particle.shape
        self.importance_weight = [np.log(np.ones(self.n_particle) / self.n_particle)] 
        self.indices = []
        self.random_state = random_state
        # print('calling thiiiiiiis ?')

    @abstractmethod
    def propagate(self,particles):
    	pass

    @abstractmethod
    def transition_density(self,at,mean,**params):
    	pass

    @abstractmethod
    def observation_density(self,obs,mean,**params):
    	pass
    
    @abstractmethod
    def importance_weight_function(self,observed):
   		pass

    @abstractmethod
    def simulation_weight_function(self,observed):
   		pass

    def multinomial_resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=normalize_log(self.simulation_weight[-1]))
        self.indices.append(index)
        return self.particle[-1][index], index


    def predict(self,observed):
    	# different with simulation weight 
        self.simulation_weight_function(observed)
        resampled, indices = self.multinomial_resample()
        propagated = self.propagate(resampled)
        self.particle.append(propagated)

        # Constant weights after resampling  (Only for bpf???)
        self.importance_weight.append(np.log(np.ones(self.n_particle) / self.n_particle))

        return indices

    def process(self, observed):
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

            indices = self.predict(obs)

            p, w = self.process(obs)

            normalized_w = normalize_log(w)

            ess_est = 1. / normalized_w.T.dot(normalized_w)

            logz_estimate = logsumexp( np.log(1./self.n_particle) + w  )

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

        return np.asarray(mean), np.asarray(cov), np.asarray(logz_estimates), np.asarray(ess), np.asarray(n_unique_sequence), np.array(w_vars)

class BPF(ParticleFilter):
    def __init__(self, **params):
        super().__init__(**params) 

        self.simulation_weight = []
        self.prev_centers_list =[]

    def importance_weight_function(self,observed):
        unnormalized = self.observation_density(obs=observed,mean=self.particle[-1],offset=self.observation_offset)
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self,observed):
        self.simulation_weight.append(self.importance_weight[-1])
        
class APF(ParticleFilter):
    def __init__(self, **params):
        super().__init__(**params) 

        self.simulation_weight = []
        self.prev_centers_list =[]

    def importance_weight_function(self,observed):
        prev_centers = self.prev_centers_list[-1]

        unnormalized = self.observation_density(obs=observed,mean=self.particle[-1],offset=self.observation_offset) - self.observation_density(obs=observed,mean=prev_centers, offset=self.observation_offset)
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self,observed):
        # prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 
        prev_centers =  np.matmul(self.particle[-1] - self.transition_offset, self.phi ) + self.transition_offset # for stoch vol. temporary 

        unnormalized = self.importance_weight[-1] + self.observation_density(obs=observed,mean=prev_centers, offset=self.observation_offset)

        self.simulation_weight.append(unnormalized)

class IAPF(ParticleFilter):

    def __init__(self, **params):
        super().__init__(**params) 

        self.simulation_weight = []
        self.prev_centers_list =[]

    def importance_weight_function(self,observed):
        prev_centers = self.prev_centers_list[-1]
        # these should already be the correct centers

        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers) 

        predictive = logmatmulexp(kernels, np.array(self.importance_weight[-1]))
        proposal = logmatmulexp(kernels, np.array(self.simulation_weight[-1]))

        lik = self.observation_density(obs=observed,mean=self.particle[-1],offset=self.observation_offset)

        unnormalized = lik + predictive - proposal
        self.importance_weight[-1] = unnormalized


    def simulation_weight_function(self,observed):

        # prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset # for lingauss
        prev_centers =  np.matmul(self.particle[-1] - self.transition_offset, self.phi ) + self.transition_offset # for stoch vol. temporary 

        kernels_at_centers = self.transition_density(at=prev_centers, mean=prev_centers) 

        pred_lik = self.observation_density(obs=observed,mean=prev_centers, offset=self.observation_offset)

        sum_numerator = logmatmulexp(kernels_at_centers, np.array(self.importance_weight[-1]))
        sum_denominator = logmatmulexp(kernels_at_centers, np.ones(kernels_at_centers.shape[0]))

        #NOVEL choice 
        # sum_numerator = logmatmulexp( logmatmulexp(kernels_at_centers, np.diag(pred_lik)), np.array(self.importance_weight[-1]))
        # sum_denominator = logmatmulexp(kernels_at_centers, pred_lik.reshape(-1,1))

        unnormalized = pred_lik + sum_numerator - sum_denominator
        self.simulation_weight.append(unnormalized)

class NewAPF(ParticleFilter):

    def __init__(self, **params):

        super().__init__(**params) 
        self.simulation_weight = []
        self.prev_centers_list = []

    def importance_weight_function(self,observed):
        prev_centers = self.prev_centers_list[-1]
        # these should already be the correct centers

        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers) 

        predictive = logmatmulexp(kernels, np.array(self.importance_weight[-1]))
        proposal = logmatmulexp(kernels, np.array(self.simulation_weight[-1]))

        lik = self.observation_density(obs=observed,mean=self.particle[-1],offset=self.observation_offset)

        unnormalized = lik + predictive - proposal
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self,observed):

        # prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 
        prev_centers =  np.matmul(self.particle[-1] - self.transition_offset, self.phi ) + self.transition_offset # for stoch vol. temporary 

        kernels_at_centers = self.transition_density(at=prev_centers, mean=prev_centers) 

        pred_lik = self.observation_density(obs=observed,mean=prev_centers, offset=self.observation_offset)

        scaled_kernels = pred_lik.reshape(-1,1) + kernels_at_centers

        logA = kernels_at_centers

        logb = logmatmulexp(scaled_kernels, np.array(self.importance_weight[-1]))

        A = np.exp(logA) + 1 * np.eye(logb.shape[0])
        b = np.exp(logb)

        # K = int(self.n_particle / 2)
        # indices_tokeep = b.argsort()[-K:][::-1]
        # unnormalized = np.zeros(b.shape)
        # smaller_b = b[indices_tokeep]
        # temp = A[:,indices_tokeep]
        # smaller_A = temp[indices_tokeep,:]

        # smallest_exp_A = np.min(smaller_A)
        # smallest_exp_b = np.min(smaller_b)
        # smallest = np.min([smallest_exp_A,smallest_exp_b])
        # smallest = np.format_float_scientific(smallest)
        # min_exp = int(re.findall(r'\d+', smallest)[-1])
        # smaller_A = smaller_A * (10**min_exp)
        # smaller_b = smaller_b * (10**min_exp)

        # res = nnls(smaller_A,smaller_b)[0]
        # np.add.at(unnormalized, indices_tokeep, res)


        unnormalized = nnls(A,b)[0]

        # if np.all(unnormalized == 0.):
        #     print('ALL zeros ... \n ')
        #     sys.exit()
        #     unnormalized = self.importance_weight[-1] #some const
        #     self.simulation_weight.append(unnormalized) # bad coding practice 
        #     return

        # if np.isinf(np.log(unnormalized)).any():
        #     print('some log inf')
        #     sys.exit()

        # if np.isnan(np.log(unnormalized)).any():
        #     print('some log  nan')
        #     sys.exit()

        to_append = np.log(unnormalized)
        self.simulation_weight.append(to_append)

        # with warnings.catch_warnings(record=True) as w:
        #     # Cause all warnings to always be triggered.
        #     warnings.simplefilter("always")
        #     # This can trigger 
        #     to_append = np.log(unnormalized)
        #     self.simulation_weight.append(to_append)
        #     if issubclass(w[-1].category, RuntimeWarning):
        #         print(to_append)

class LinearGaussianPF(ParticleFilter):
    def __init__(self,init_particle,random_state, transition_cov, observation_cov, transition_mat, transition_offset, observation_mat, observation_offset):

        super().__init__(init_particle,random_state) 

        self.transition_cov = transition_cov
        self.observation_cov = observation_cov
        self.transition_offset = transition_offset
        self.observation_offset = observation_offset
        self.transition_mat = transition_mat
        self.observation_mat = observation_mat

    def transition_density(self,at,mean,**params):

        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(self.transition_cov)
        at = np.expand_dims(at,axis=1)
        at = torch.from_numpy(at)

        cov_all = cov[None, ...].repeat_interleave(self.n_particle, 0)

        kernels = MultivariateNormal(mean, cov_all).log_prob(at)

        return kernels.numpy()

    def observation_density(self,obs,mean,**params):

        mean_all = np.matmul(np.array(mean),self.observation_mat) + params['offset']
        obs = torch.from_numpy(obs)
        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)

        mean_all = torch.from_numpy(mean_all)
        obs_cov = torch.from_numpy(self.observation_cov)

        liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()

    def propagate(self, resmpled_particles):
        prev_centers = resmpled_particles.dot(self.transition_mat) + self.transition_offset 
        # need these in APF for importance weight computation 
        self.prev_centers_list.append(prev_centers)
        res = prev_centers + np.random.multivariate_normal(mean=np.zeros(self.ndim_hidden),cov=self.transition_cov,size=self.n_particle)
        return res

class LinearGaussianBPF(BPF, LinearGaussianPF):
    def __init__(self,init_particle,random_state, transition_cov, observation_cov, transition_mat, transition_offset, observation_mat, observation_offset):
        super(LinearGaussianBPF, self).__init__(init_particle=init_particle,
                                            random_state=random_state, 
                                            transition_cov=transition_cov, 
                                            observation_cov=observation_cov, 
                                            transition_mat=transition_mat, 
                                            transition_offset=transition_offset, 
                                            observation_mat=observation_mat, 
                                            observation_offset=observation_offset)

class LinearGaussianAPF(APF, LinearGaussianPF):
    def __init__(self,init_particle,random_state, transition_cov, observation_cov, transition_mat, transition_offset, observation_mat, observation_offset):
        super(LinearGaussianAPF, self).__init__(init_particle=init_particle,
                                            random_state=random_state, 
                                            transition_cov=transition_cov, 
                                            observation_cov=observation_cov, 
                                            transition_mat=transition_mat, 
                                            transition_offset=transition_offset, 
                                            observation_mat=observation_mat, 
                                            observation_offset=observation_offset)

class LinearGaussianIAPF(IAPF, LinearGaussianPF):
    def __init__(self,init_particle,random_state, transition_cov, observation_cov, transition_mat, transition_offset, observation_mat, observation_offset):
        super(LinearGaussianIAPF, self).__init__(init_particle=init_particle,
                                            random_state=random_state, 
                                            transition_cov=transition_cov, 
                                            observation_cov=observation_cov, 
                                            transition_mat=transition_mat, 
                                            transition_offset=transition_offset, 
                                            observation_mat=observation_mat, 
                                            observation_offset=observation_offset)

class LinearGaussianNewAPF(NewAPF, LinearGaussianPF):
    def __init__(self,init_particle,random_state, transition_cov, observation_cov, transition_mat, transition_offset, observation_mat, observation_offset):
        super(LinearGaussianNewAPF, self).__init__(init_particle=init_particle,
                                            random_state=random_state, 
                                            transition_cov=transition_cov, 
                                            observation_cov=observation_cov, 
                                            transition_mat=transition_mat, 
                                            transition_offset=transition_offset, 
                                            observation_mat=observation_mat, 
                                            observation_offset=observation_offset)



class StochVolPF(ParticleFilter):
    def __init__(self,init_particle,random_state, transition_cov, transition_offset, phi):
        super().__init__(init_particle,random_state) 
        self.transition_cov = transition_cov
        self.transition_offset = transition_offset
        self.phi = phi
        self.observation_offset = np.zeros(self.ndim_hidden,)

    def transition_density(self,at,mean,**params):

        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(self.transition_cov)

        at = np.expand_dims(at,axis=1)
        at = torch.from_numpy(at)

        cov_all = cov[None, ...].repeat_interleave(self.n_particle, 0)

        kernels = MultivariateNormal(mean, cov_all).log_prob(at)

        return kernels.numpy()

    def observation_density(self,obs,mean,**params):

        actual_mean = np.zeros((self.n_particle,self.ndim_hidden))

        obs = torch.from_numpy(obs).double()

        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)

        # obs_all = obs_all[:,None,:]

        actual_mean_all = torch.from_numpy(actual_mean).double()

        obs_cov = np.exp(mean)

        obs_cov = torch.from_numpy(np.apply_along_axis(np.diag,1,obs_cov)).double()

        liks = MultivariateNormal(actual_mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()

    def propagate(self, resampled_particles):
        prev_centers =  np.matmul(resampled_particles - self.transition_offset, self.phi ) + self.transition_offset
        # need these in APF for importance weight computation 
        self.prev_centers_list.append(prev_centers)
        res = prev_centers + np.random.multivariate_normal(mean=np.zeros(self.ndim_hidden),cov=self.transition_cov,size=self.n_particle)
        return res

class StochVolBPF(BPF, StochVolPF):
    def __init__(self,init_particle,random_state, transition_cov, transition_offset, phi):
        super(StochVolBPF, self).__init__(init_particle=init_particle,
                                            random_state=random_state, 
                                            transition_cov=transition_cov, 
                                            transition_offset=transition_offset,
                                            phi=phi)

class StochVolAPF(APF, StochVolPF):
    def __init__(self,init_particle,random_state, transition_cov, transition_offset, phi):
        super(StochVolAPF, self).__init__(init_particle=init_particle,
                                            random_state=random_state, 
                                            transition_cov=transition_cov, 
                                            transition_offset=transition_offset,
                                            phi=phi)

class StochVolIAPF(IAPF, StochVolPF):
    def __init__(self,init_particle,random_state, transition_cov, transition_offset, phi):
        super(StochVolIAPF, self).__init__(init_particle=init_particle,
                                            random_state=random_state, 
                                            transition_cov=transition_cov, 
                                            transition_offset=transition_offset,
                                            phi=phi)

class StochVolNewAPF(NewAPF, StochVolPF):
    def __init__(self,init_particle,random_state, transition_cov, transition_offset, phi):
        super(StochVolNewAPF, self).__init__(init_particle=init_particle,
                                            random_state=random_state, 
                                            transition_cov=transition_cov, 
                                            transition_offset=transition_offset,
                                            phi=phi)

