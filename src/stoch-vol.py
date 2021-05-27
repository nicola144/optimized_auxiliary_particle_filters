

from particles import *
from utils import *


set_plotting()
dim = 10
timesteps = 100

constant_mean = np.zeros(dim, )
initial_cov = np.eye(dim)
varcov = 2.5
# varphi = .5
transition_cov = varcov * np.eye(dim)
# phi = varphi * np.diag(np.ones(dim, ))
phi = np.ones((dim,dim)) * 1/(dim)

def prior_sample(size=1):
	res = np.random.multivariate_normal(mean=constant_mean,cov=initial_cov,size=size)
	return res.squeeze()

def transition_sample(prev_state):
	return np.random.multivariate_normal(mean=constant_mean + np.dot(phi, prev_state - constant_mean), cov=transition_cov)

def observation_sample(curr_state):
	obs_cov = np.diag(np.exp(curr_state))

	assert np.all( np.linalg.eigh( obs_cov )[0] > 0 )
	return np.random.multivariate_normal(mean=np.zeros(dim,), cov=obs_cov)



seeds = np.loadtxt('seeds.out').astype('int64')

allparticles_ess_bpf = []
allparticles_ess_apf = []
allparticles_ess_iapf = []
allparticles_ess_oapf = []

allparticles_joint_logliks_bpf = []
allparticles_joint_logliks_apf = []
allparticles_joint_logliks_iapf = []
allparticles_joint_logliks_oapf = []

for n_particle in tqdm([100,1000]):

	all_ess_bpf = []
	all_ess_apf = []
	all_ess_iapf = []
	all_ess_oapf = []

	all_joint_logliks_bpf = []
	all_joint_logliks_apf = []
	all_joint_logliks_iapf = []
	all_joint_logliks_oapf = []

	for seed in tqdm(seeds):

		random_state = np.random.RandomState(seed)

		# Data generation

		obs = []
		states = []

		curr_state = prior_sample()

		states.append(curr_state)

		obs.append( observation_sample(curr_state) )

		for t in range(timesteps - 1):
			# transition state
			curr_state = transition_sample(curr_state)

			states.append(curr_state)

			#get obs
			obs.append(observation_sample(curr_state))


		observations = np.array(obs)
		states = np.array(states)

		# End of data gen.

		bpf = StochVolBPF(init_particle=prior_sample(size=n_particle),
								random_state=random_state,
								transition_cov=transition_cov,
								transition_offset=constant_mean,
								phi=phi )

		apf = StochVolAPF(init_particle=prior_sample(size=n_particle),
								random_state=random_state,
								transition_cov=transition_cov,
								transition_offset=constant_mean,
								phi=phi )

		iapf = StochVolIAPF(init_particle=prior_sample(size=n_particle),
								random_state=random_state,
								transition_cov=transition_cov,
								transition_offset=constant_mean,
								phi=phi )

		oapf = StochVolOAPF(init_particle=prior_sample(size=n_particle),
								random_state=random_state,
								transition_cov=transition_cov,
								transition_offset=constant_mean,
								phi=phi )


		mean_bpf, covs_bpf, ess_bpf, n_unique_bpf, w_vars_bpf, liks_bpf, joint_liks_bpf, times_bpf = bpf.filter(observations)
		mean_apf, covs_apf, ess_apf, n_unique_apf, w_vars_apf, liks_apf, joint_liks_apf, times_apf = apf.filter(observations)
		mean_iapf, covs_iapf, ess_iapf, n_unique_iapf, w_vars_iapf, liks_iapf, joint_liks_iapf, times_iapf = iapf.filter(observations)
		mean_oapf, covs_oapf, ess_oapf, n_unique_oapf, w_vars_oapf, liks_oapf, joint_liks_oapf, times_oapf = oapf.filter(observations)

		all_ess_bpf.append(ess_bpf)
		all_ess_apf.append(ess_apf)
		all_ess_iapf.append(ess_iapf)
		all_ess_oapf.append(ess_oapf)

		all_joint_logliks_bpf.append(joint_liks_bpf)
		all_joint_logliks_apf.append(joint_liks_apf)
		all_joint_logliks_iapf.append(joint_liks_iapf)
		all_joint_logliks_oapf.append(joint_liks_oapf)

		# f,ax = plt.subplots(mean_bpf.shape[1],5)
		# for i in range(mean_bpf.shape[1]):
		#
		# 	ax[i][0].plot(mean_bpf, 'b')
		# 	ax[i][1].plot(mean_apf, 'y')
		# 	ax[i][2].plot(mean_iapf, 'c')
		# 	ax[i][3].plot(mean_oapf, 'm')
		# 	ax[i][4].plot(states, 'k')
		# plt.tight_layout()
		# plt.show()
		# exit()

	allparticles_joint_logliks_bpf.append(all_joint_logliks_bpf)
	allparticles_joint_logliks_apf.append(all_joint_logliks_apf)
	allparticles_joint_logliks_iapf.append(all_joint_logliks_iapf)
	allparticles_joint_logliks_oapf.append(all_joint_logliks_oapf)

	allparticles_ess_bpf.append(all_ess_bpf)
	allparticles_ess_apf.append(all_ess_apf)
	allparticles_ess_iapf.append(all_ess_iapf)
	allparticles_ess_oapf.append(all_ess_oapf)

res_ess = np.vstack([
	allparticles_ess_bpf,
	allparticles_ess_apf,
	all_ess_iapf,
	all_ess_oapf
])

res_liks = np.vstack([
	allparticles_joint_logliks_bpf.pop(),
	allparticles_joint_logliks_apf.pop(),
	allparticles_joint_logliks_iapf.pop(),
	allparticles_joint_logliks_oapf.pop()
])




print("Settings")
print('timesteps', timesteps)
print('dim', dim)
print('varcov', varcov)
# print('vaphi',varphi)


np.savetxt('results/stochvol/joint_logliks/moreparticles_results_stochvol_reduced5_jointliks-seq-dim'+str(dim)+ 'varcov-'+ str(varcov) + '-varphi_corr_oneovertwo.out', res_liks, delimiter=',')

# np.savetxt('results/stochvol/results_stochvol_wvar_'+str(n_particle)+'_particles-dim'+str(dim)+'.out', res_wvars, delimiter=',')
# np.savetxt('results/stochvol/results_stochvol_nunique_'+str(n_particle)+'_particles-dim'+str(dim)+'.out', res_nunique, delimiter=',')


# Plotting







