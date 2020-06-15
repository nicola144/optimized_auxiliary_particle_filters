import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal as multivariate_normal_scipy
import sys
from tqdm import tqdm
from scipy.optimize import nnls,linprog
from utils import *

np.random.seed(0)
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

        # Constant weights after resampling 
        self.importance_weight.append(np.ones(self.n_particle) / self.n_particle)
        # return propagated, self.importance_weight[-1]

    def process(self, observed):
        self.importance_weight_function(observed)
        return self.particle[-1], self.importance_weight[-1]

    def filter(self, observed_sequence):
        mean = []
        cov = []
        for obs in tqdm(observed_sequence):
            self.predict(obs)
            p, w = self.process(obs)
            if np.isnan(w).any():
                print('some weights are nan')
                sys.exit()

            mean.append(np.average(p, axis=0, weights=w))
            cov.append(np.cov(p, rowvar=False, aweights=w))
        return np.asarray(mean), np.asarray(cov)

class BPF(ParticleFilter):
    def __init__(self,init_particle,random_state):
        super().__init__(init_particle,random_state) 
    
    def importance_weight_function(self,observed):
        unnormalized = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset)
        self.importance_weight[-1] = normalize(unnormalized)

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
        unnormalized = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset) / self.observation_density(obs=observed,mean=prev_centers, offset=self.observation_offset)
        self.importance_weight[-1] = normalize(unnormalized)

    def simulation_weight_function(self,observed):
        prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 
        prev_centers = prev_centers[np.newaxis,...]
        unnormalized = self.importance_weight[-1] * self.observation_density(obs=observed,mean=prev_centers.tolist(), offset=self.observation_offset)

        self.simulation_weight.append(normalize(unnormalized))

    def multinomial_resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=self.simulation_weight[-1])
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

        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers[-1]) #.T 
        # print(kernels.shape)
        # sys.exit()
        # predictive = kernels.dot(self.importance_weight[-1])
        predictive = np.matmul(kernels, self.importance_weight[-1])
        lik = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset)
        # proposal = kernels.T.dot(self.simulation_weight[-1])
        proposal = np.matmul(kernels, self.simulation_weight[-1])
        unnormalized =  ( lik * predictive) / ( proposal )
        self.importance_weight[-1] = normalize(unnormalized)

    def simulation_weight_function(self,observed):
        prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 
        prev_centers = prev_centers[np.newaxis,...]

        kernels_at_centers = self.transition_density(at=prev_centers[-1], mean=prev_centers[-1]) #.T 
        # sum_numerator = kernels_at_centers.T.dot(self.importance_weight[-1])
        # sum_denominator = kernels_at_centers.T.dot(np.ones(kernels_at_centers.shape[0]))
        sum_numerator = np.matmul(kernels_at_centers, self.importance_weight[-1])
        sum_denominator = np.matmul(kernels_at_centers, np.ones(kernels_at_centers.shape[0]))
        pred_lik = self.observation_density(obs=observed,mean=prev_centers.tolist(), offset=self.observation_offset)
        unnormalized = (pred_lik * sum_numerator) / sum_denominator
        # print(unnormalized.shape)
        # sys.exit()
        self.simulation_weight.append(normalize(unnormalized))

    def multinomial_resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=self.simulation_weight[-1])
        return self.particle[-1][index]

class NewAPF(ParticleFilter):
    def __init__(self,init_particle,random_state):
        super().__init__(init_particle,random_state)
        self.simulation_weight = []
        self.prev_centers = []

    def importance_weight_function(self,observed):
        prev_centers = self.prev_centers[-1]
        # prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 

        prev_centers = prev_centers[np.newaxis,...]

        # these should already be the correct centers

        # unnormalized = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset) / self.observation_density(obs=observed,mean=prev_centers, offset=self.observation_offset)
        kernels = self.transition_density(at=self.particle[-1], mean=prev_centers[-1]) # defo should be a row matrix 

        predictive = np.matmul(kernels, self.importance_weight[-1])
        proposal = np.matmul(kernels, self.simulation_weight[-1])

        lik = self.observation_density(obs=observed,mean=self.particle,offset=self.observation_offset)
        unnormalized =  ( lik * predictive) / ( proposal )
        self.importance_weight[-1] = normalize(unnormalized)


    def simulation_weight_function(self,observed):
        prev_centers = self.particle[-1].dot(self.transition_mat) + self.transition_offset 
        prev_centers = prev_centers[np.newaxis,...]

        kernels_at_centers = self.transition_density(at=prev_centers[-1], mean=prev_centers[-1]) # it doesnt acutally matter .T since symmetric 

        pred_lik = self.observation_density(obs=observed,mean=prev_centers.tolist(), offset=self.observation_offset)

        scaled_kernels = pred_lik.reshape(-1,1) * kernels_at_centers
        # scaled_kernels = self.scaled_transition_density(at=prev_centers[-1], mean=prev_centers[-1],observed=observed)


        A = kernels_at_centers  

        # A[np.abs(A) < 1e-9] = 0.
        # b = scaled_kernels.T.dot(self.importance_weight[-1])
        # b = np.dot(scaled_kernels,self.importance_weight[-1])

        b = np.matmul(scaled_kernels,self.importance_weight[-1])

        # print(A)
        # print(b)
        # sys.exit()


        unnormalized = nnls(A,b)[0]


        # A_ub =  np.array( [ np.concatenate(( -np.ones(b.shape[0]), np.zeros(b.shape[0])  )) ] )
        # b_ub = [- 1./self.n_particle]


        # A = np.hstack((A, -np.eye(b.shape[0])))
        # c = np.concatenate(( np.zeros(b.shape[0]), np.ones(b.shape[0])  ))
        # results = linprog(c=c, A_eq=A, b_eq=b, A_ub=A_ub,b_ub=b_ub, bounds=[(0,None)]*b.shape[0]*2, method='interior-point',options={'presolve':True,'disp':False,'sparse':True}) # ,options={'presolve':False} can be interior-point or revised simplex
        # result = "\n Success! \n" if results['status'] == 0 else "\n Something went wrong :( \n " 
        # print(result)
        # result_vec = results['x']
        # unnormalized = result_vec[:b.shape[0]]

        if not np.all(unnormalized >= 0. ):
            print("some negative")
            print(unnormalized)
            sys.exit()
        if not unnormalized.shape == (self.n_particle,): 
            print('wrong shape')
            sys.exit()

        if np.all(unnormalized == 0.):
            print('ALL zeros ... \n ')
            sys.exit()
        
        self.simulation_weight.append(normalize(unnormalized))

    def multinomial_resample(self):
        if np.isnan(self.simulation_weight[-1]).any():
            print('some preweights nan')
            sys.exit()

        index = np.random.choice(self.n_particle, self.n_particle, p=self.simulation_weight[-1])
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
 
        kernels = np.vstack(( [np.array([  multivariate_normal_scipy.pdf(at[j], mean=mean[l], cov=self.transition_cov) for l in range(self.n_particle) ])] for j in range(self.n_particle) ))

        return kernels

    # def scaled_transition_density(self,at,mean,observed):
    #     kernels = np.empty((self.n_particle,self.n_particle))
    #     for j in range(self.n_particle):
    #         for l in range(self.n_particle):
    #             pred_lik = multivariate_normal_scipy.pdf(observed, mean=np.dot(self.observation_mat, at[j]) + self.observation_offset, cov=self.observation_cov  )
    #             kernels[j][l] =  pred_lik * multivariate_normal_scipy.pdf(at[j], mean=mean[l], cov=self.transition_cov)
    #     return kernels

    def observation_density(self,obs,mean,**params):
        liks = []
        for m in range(self.n_particle):
            # print(np.array(mean[-1][m]).shape)
            liks.append(multivariate_normal_scipy.pdf(obs, mean=np.dot(self.observation_mat, mean[-1][m]) + params['offset'], cov=self.observation_cov))
        return np.array(liks)

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
        kernels = np.vstack(( [np.array([  multivariate_normal_scipy.pdf(at[j], mean=mean[l], cov=self.transition_cov) for l in range(self.n_particle) ])] for j in range(self.n_particle) ))
        return np.array(kernels)

    def observation_density(self,obs,mean,**params):
        liks = []
        for m in range(self.n_particle):
            liks.append(multivariate_normal_scipy.pdf(obs, mean=np.dot(self.observation_mat, mean[-1][m]) + params['offset'], cov=self.observation_cov))
        return np.array(liks)

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
        liks = []
        for m in range(self.n_particle):
            liks.append(multivariate_normal_scipy.pdf(obs, mean=np.dot(self.observation_mat, mean[-1][m]) + params['offset'], cov=self.observation_cov))
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
    def transition_density(self,at,mean,**params):
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











