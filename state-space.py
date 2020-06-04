import numpy as np
from scipy.special import logsumexp



class SMC():

	def __init__(self, init_particle, transition, *transition_args, nll):
		"""
        Parameters
        ----------
        init_particle: n_particle, ndim_hidden) np.ndarray
        			   initial hidden state




		"""
        self.particle = [init_particle]
        self.n_particle, self.ndim_hidden = init_particle.shape
        self.weight = [np.ones(self.n_particle) / self.n_particle]
        self.transition = transition
        self.nll = nll


    def resample(self):
        index = np.random.choice(self.n_particle, self.n_particle, p=self.weight[-1])
        return self.particle[-1][index]


    def predict(self):
    	predicted = self.transition(self.resample(), *transition_args)
        self.particle.append(predicted)
        self.weight.append(np.ones(self.n_particle) / self.n_particle)
        return predicted, self.weight[-1]

    def weigh(self, observed):
        logit = -self.nll(observed, self.particle[-1])
        logit -= logsumexp(logit)
        self.weight[-1] = np.exp(logit)


