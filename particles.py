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
        # self.importance_weight = [np.ones(self.n_particle) / self.n_particle]
        self.importance_weight = [np.log(np.ones(self.n_particle) / self.n_particle)]

        self.random_state = random_state

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

    @abstractmethod
    def multinomial_resample(self):
    	pass

    def predict(self,observed):
    	# different with simulation weight 
        self.simulation_weight_function(observed)
        resampled = self.multinomial_resample()
        propagated = self.propagate(resampled)
        self.particle.append(propagated)

        # Constant weights after resampling  (Only for bpf???)
        self.importance_weight.append(np.log(np.ones(self.n_particle) / self.n_particle))

    def process(self, observed):
        self.importance_weight_function(observed)
        return self.particle[-1], self.importance_weight[-1]

    def filter(self, observed_sequence):
        mean = []
        cov = []
        logz_estimates = []
        ess = []

        for obs in tqdm(observed_sequence):
            self.predict(obs)
            p, w = self.process(obs)
            normalized_w = normalize_log(w)
            ess_est = 1. / normalized_w.T.dot(normalized_w)
            logz_estimate = logsumexp( np.log(1./self.n_particle) + w  )

            if np.isnan(normalized_w).any():
                print('some weights are nan')
                sys.exit()

            mean.append(np.average(p, axis=0, weights=normalized_w))
            cov.append(np.cov(p, rowvar=False, aweights=normalized_w))
            ess.append(ess_est)
            logz_estimates.append(logz_estimate)

        return np.asarray(mean), np.asarray(cov), np.asarray(logz_estimates), np.asarray(ess)

class BPF(ParticleFilter):
    def __init__(self,init_particle,random_state):
        super().__init__(init_particle,random_state) 
        self.simulation_weight = []
    
    def importance_weight_function(self,observed):

        # I guess ? 
        # self.importance_weight.append(np.log(np.ones(self.n_particle) / self.n_particle))

        unnormalized = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset)
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self,observed):
        self.simulation_weight.append(self.importance_weight[-1])
        
    def multinomial_resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=normalize_log(self.simulation_weight[-1]))
        return self.particle[-1][index]

class APF(ParticleFilter):
    def __init__(self,init_particle,random_state):
        super().__init__(init_particle,random_state)
        self.simulation_weight = []
        self.prev_centers =[]

    def importance_weight_function(self,observed):
        prev_centers = self.prev_centers[-1]
        prev_centers = prev_centers[np.newaxis,...]
        # unnormalized = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset) / self.observation_density(obs=observed,mean=prev_centers, offset=self.observation_offset)
        # self.importance_weight[-1] = normalize(unnormalized)
        unnormalized = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset) - self.observation_density(obs=observed,mean=prev_centers, offset=self.observation_offset)
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self,observed):
        prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 
        prev_centers = prev_centers[np.newaxis,...]
        unnormalized = self.importance_weight[-1] + self.observation_density(obs=observed,mean=prev_centers.tolist(), offset=self.observation_offset)

        # self.simulation_weight.append(normalize(unnormalized))
        self.simulation_weight.append(unnormalized)


    def multinomial_resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=normalize_log(self.simulation_weight[-1]))
        return self.particle[-1][index]

class IAPF(ParticleFilter):

    def __init__(self,init_particle,random_state):
        super().__init__(init_particle,random_state)
        self.simulation_weight = []
        self.prev_centers =[]

    def importance_weight_function(self,observed):
        prev_centers = self.prev_centers[-1]
        prev_centers = prev_centers[np.newaxis,...]
        # these should already be the correct centers

        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers[-1]) 

        predictive = logmatmulexp(kernels, np.array(self.importance_weight[-1]))
        proposal = logmatmulexp(kernels, np.array(self.simulation_weight[-1]))

        lik = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset)

        unnormalized = lik + predictive - proposal
        self.importance_weight[-1] = unnormalized


    def simulation_weight_function(self,observed):
        prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 
        prev_centers = prev_centers[np.newaxis,...]

        kernels_at_centers = self.transition_density(at=prev_centers[-1], mean=prev_centers[-1]) 

        pred_lik = self.observation_density(obs=observed,mean=prev_centers.tolist(), offset=self.observation_offset)

        sum_numerator = logmatmulexp(kernels_at_centers, np.array(self.importance_weight[-1]))
        sum_denominator = logmatmulexp(kernels_at_centers, np.ones(kernels_at_centers.shape[0]))

        #NOVEL choice 
        # sum_numerator = logmatmulexp( logmatmulexp(kernels_at_centers, np.diag(pred_lik)), np.array(self.importance_weight[-1]))
        # sum_denominator = logmatmulexp(kernels_at_centers, pred_lik.reshape(-1,1))

        unnormalized = pred_lik + sum_numerator - sum_denominator
        self.simulation_weight.append(unnormalized)


    def multinomial_resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=normalize_log(self.simulation_weight[-1]))
        return self.particle[-1][index]

class NewAPF(ParticleFilter):
    def __init__(self,init_particle,random_state):
        super().__init__(init_particle,random_state)
        self.simulation_weight = []
        self.prev_centers = []

    def importance_weight_function(self,observed):
        prev_centers = self.prev_centers[-1]
        prev_centers = prev_centers[np.newaxis,...]
        # these should already be the correct centers

        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers[-1]) 

        predictive = logmatmulexp(kernels, np.array(self.importance_weight[-1]))
        proposal = logmatmulexp(kernels, np.array(self.simulation_weight[-1]))

        lik = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset)

        unnormalized = lik + predictive - proposal
        self.importance_weight[-1] = unnormalized

    def simulation_weight_function(self,observed):

        prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 
        prev_centers = prev_centers[np.newaxis,...]

        kernels_at_centers = self.transition_density(at=prev_centers[-1], mean=prev_centers[-1]) # it doesnt acutally matter .T since symmetric 

        pred_lik = self.observation_density(obs=observed,mean=prev_centers.tolist(), offset=self.observation_offset)

        scaled_kernels = pred_lik.reshape(-1,1) + kernels_at_centers

        logA = kernels_at_centers

        logb = logmatmulexp(scaled_kernels, np.array(self.importance_weight[-1]))

        A = np.exp(logA)
        b = np.exp(logb)

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

        self.simulation_weight.append(unnormalized)

    def multinomial_resample(self):

        index = np.random.choice(self.n_particle, self.n_particle, p=normalize(self.simulation_weight[-1]))

        return self.particle[-1][index]

class LinearGaussianNewAPF(NewAPF):
    def __init__(self,init_particle,random_state, transition_cov, observation_cov, transition_mat, transition_offset, observation_mat, observation_offset):
        super().__init__(init_particle,random_state) 
        self.transition_cov = transition_cov
        self.observation_cov = observation_cov
        self.transition_offset = transition_offset
        self.observation_offset = observation_offset
        self.transition_mat = transition_mat
        self.observation_mat = observation_mat

    def transition_density(self,at,mean,**params):
        # kernels = np.empty((self.n_particle,self.n_particle))
        # for j in range(self.n_particle):
        #     for l in range(self.n_particle):
        #         kernels[j][l] = multivariate_normal_scipy.pdf(at[j], mean=mean[l], cov=self.transition_cov)
 
        # kernels = np.vstack(( [np.array([  multivariate_normal_scipy.logpdf(at[j], mean=mean[l], cov=self.transition_cov) for l in range(self.n_particle) ])] for j in range(self.n_particle) ))

        mean = torch.from_numpy(mean)
        cov = torch.from_numpy(self.transition_cov)
        at = np.expand_dims(at,axis=1)
        at = torch.from_numpy(at)

        cov_all = cov[None, ...].repeat_interleave(self.n_particle, 0)

        kernels = MultivariateNormal(mean, cov_all).log_prob(at)

        return kernels.numpy()

    def observation_density(self,obs,mean,**params):
        mean_all = np.matmul(np.array(mean[-1]),self.observation_mat) + params['offset']
        obs = torch.from_numpy(obs)
        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)

        mean_all = torch.from_numpy(mean_all)
        obs_cov = torch.from_numpy(self.observation_cov)

        liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()

    def propagate(self, particles):
        prev_centers = particles.dot(self.transition_mat) + self.transition_offset 
        # need these in APF for importance weight computation 
        self.prev_centers.append(prev_centers)
        res = prev_centers + np.random.multivariate_normal(mean=np.zeros(self.ndim_hidden),cov=self.transition_cov,size=self.n_particle)
        return res

class LinearGaussianIAPF(IAPF):
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

        # kernels = np.vstack(( [np.array([  multivariate_normal_scipy.logpdf(at[j], mean=mean[l], cov=self.transition_cov) for l in range(self.n_particle) ])] for j in range(self.n_particle) ))
        
        return kernels.numpy()

    def observation_density(self,obs,mean,**params):

        mean_all = np.matmul(np.array(mean[-1]),self.observation_mat) + params['offset']
        obs = torch.from_numpy(obs).double()
        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)

        mean_all = torch.from_numpy(mean_all).double()
        obs_cov = torch.from_numpy(self.observation_cov).double()

        liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()

        # liks2 = []
        # for m in range(self.n_particle):
        #     liks2.append(multivariate_normal_scipy.logpdf(obs, mean=np.dot(self.observation_mat, mean[-1][m]) + params['offset'], cov=self.observation_cov))

        # return np.array(liks2)

    def propagate(self, particles):
        prev_centers = particles.dot(self.transition_mat) + self.transition_offset 
        # need these in APF for importance weight computation 
        self.prev_centers.append(prev_centers)
        res = prev_centers + np.random.multivariate_normal(mean=np.zeros(self.ndim_hidden),cov=self.transition_cov,size=self.n_particle)
        return res

class LinearGaussianAPF(APF):

    def __init__(self,init_particle,random_state, transition_cov, observation_cov, transition_mat, transition_offset, observation_mat, observation_offset):
        super().__init__(init_particle,random_state) 
        self.transition_cov = transition_cov
        self.observation_cov = observation_cov
        self.transition_offset = transition_offset
        self.observation_offset = observation_offset
        self.transition_mat = transition_mat
        self.observation_mat = observation_mat

    # not needed to evaluate in APF
    def transition_density(self,at,mean,**params):
        pass

    def observation_density(self,obs,mean,**params):
        # liks = []
        # for m in range(self.n_particle):
        #     liks.append(multivariate_normal_scipy.logpdf(obs, mean=np.dot(self.observation_mat, mean[-1][m]) + params['offset'], cov=self.observation_cov))
        # return np.array(liks)

        mean_all = np.matmul(np.array(mean[-1]),self.observation_mat) + params['offset']
        obs = torch.from_numpy(obs).double()
        obs_all = obs[None, ...].repeat_interleave(self.n_particle, 0)

        mean_all = torch.from_numpy(mean_all).double()
        obs_cov = torch.from_numpy(self.observation_cov).double()

        liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()


    def propagate(self, particles):
        prev_centers = particles.dot(self.transition_mat) + self.transition_offset 
        # need these in APF for importance weight computation 
        self.prev_centers.append(prev_centers)
        res = prev_centers + np.random.multivariate_normal(mean=np.zeros(self.ndim_hidden),cov=self.transition_cov,size=self.n_particle)
        return res

class LinearGaussianBPF(BPF):
    
    def __init__(self,init_particle,random_state, transition_cov, observation_cov, transition_mat, transition_offset, observation_mat, observation_offset):
        super().__init__(init_particle,random_state) 
        self.transition_cov = transition_cov
        self.observation_cov = observation_cov
        self.transition_offset = transition_offset
        self.observation_offset = observation_offset
        self.transition_mat = transition_mat
        self.observation_mat = observation_mat

    # not needed to evaluate in BPF
    def transition_density(self,at,mean,**params):
        pass

    def observation_density(self,obs,mean,**params):
        # liks = []
        # for m in range(self.n_particle):
        #     liks.append(multivariate_normal_scipy.logpdf(obs, mean= np.dot(self.observation_mat, mean[-1][m]) + params['offset'], cov=self.observation_cov))
        # return np.array(liks)
        mean_all = np.matmul(np.array(mean[-1]),self.observation_mat) + params['offset']
        obs_all = np.array( [[obs],] * self.n_particle ).squeeze()

        mean_all = torch.from_numpy(mean_all).double()
        obs_all = torch.from_numpy(obs_all).double()
        obs_cov = torch.from_numpy(self.observation_cov).double()

        liks = MultivariateNormal(mean_all, obs_cov).log_prob(obs_all)

        return liks.numpy()

    def propagate(self, particles):
        prev_centers = particles.dot(self.transition_mat) + self.transition_offset 
        res = prev_centers + np.random.multivariate_normal(mean=np.zeros(self.ndim_hidden),cov=self.transition_cov,size=self.n_particle)
        return res











