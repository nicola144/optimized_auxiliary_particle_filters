import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal as multivariate_normal_scipy
import sys
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
        self.importance_weight = [np.ones(self.n_particle) / self.n_particle]
        self.random_state = random_state

    @abstractmethod
    def propagate(self,particles):
    	pass

    @abstractmethod
    def transition_density(self,at):
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

        # Constant weights after resampling 
        self.importance_weight.append(np.ones(self.n_particle) / self.n_particle)
        # return propagated, self.importance_weight[-1]

    def process(self, observed):
        self.importance_weight_function(observed)
        return self.particle[-1], self.importance_weight[-1]

    def filter(self, observed_sequence):
        mean = []
        cov = []
        for obs in observed_sequence:
            self.predict(obs)
            p, w = self.process(obs)
            mean.append(np.average(p, axis=0, weights=w))
            cov.append(np.cov(p, rowvar=False, aweights=w))
        return np.asarray(mean), np.asarray(cov)


class BPF(ParticleFilter):
    def __init__(self,init_particle,random_state):
        super().__init__(init_particle,random_state) 
    
    def importance_weight_function(self,observed):
        unnormalized = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset)
        self.importance_weight[-1] = unnormalized / np.sum(unnormalized)

    # not needed in BPF
    def simulation_weight_function(self,observed):
        pass
        
    def multinomial_resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=self.importance_weight[-1])
        return self.particle[-1][index]

class APF(ParticleFilter):
    def __init__(self,init_particle,random_state):
        super().__init__(init_particle,random_state)
        self.simulation_weight = []
        self.prev_centers =[]

    def importance_weight_function(self,observed):
        prev_centers = self.prev_centers[-1]
        prev_centers = prev_centers[np.newaxis,...]
        unnormalized = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset) / self.observation_density(obs=observed,mean=prev_centers, offset=self.observation_offset )
        self.importance_weight[-1] = unnormalized / np.sum(unnormalized)

    def simulation_weight_function(self,observed):
        prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 
        prev_centers = prev_centers[np.newaxis,...]
        unnormalized = self.importance_weight[-1] * self.observation_density(obs=observed,mean=prev_centers.tolist(), offset=self.observation_offset )
        self.simulation_weight.append(unnormalized / np.sum(unnormalized))

    def multinomial_resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=self.simulation_weight[-1])
        return self.particle[-1][index]


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
    def transition_density(self,at):
        pass

    def observation_density(self,obs,mean,**params):
        liks = []
        for m in range(self.n_particle):
            liks.append(multivariate_normal_scipy.pdf(obs, np.dot(self.observation_mat, mean=mean[-1][m]) + params['offset'], cov=self.observation_cov))
        return np.array(liks)

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
    def transition_density(self,at):
        pass

    def observation_density(self,obs,mean,**params):
        liks = []
        for m in range(self.n_particle):
            liks.append(multivariate_normal_scipy.pdf(obs, mean= np.dot(self.observation_mat, mean[-1][m]) + params['offset'], cov=self.observation_cov))
        return np.array(liks)

    def propagate(self, particles):
        prev_centers = particles.dot(self.transition_mat) + self.transition_offset 
        res = prev_centers + np.random.multivariate_normal(mean=np.zeros(self.ndim_hidden),cov=self.transition_cov,size=self.n_particle)
        return res











