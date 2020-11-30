
from particles import *
from utils import *

dim = 2
timesteps = 100
n_particle = 100

constant_mean = np.zeros(dim, )
initial_cov = np.eye(dim)
transition_cov = 1 * np.eye(dim)
phi = np.diag(np.ones(dim, ))


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


# set_plotting()

# for more plots
# fig, ax = plt.subplots(1, 2)
# ax = ax.reshape(1,2)
# fig.tight_layout(pad=0.3)
# plt.subplots_adjust(top=0.95)
# plt.tight_layout(pad=0.3)



all_ess_bpf = []
all_ess_apf = []
all_ess_iapf = []
all_ess_oapf = []

all_wvar_bpf = []
all_wvar_apf = []
all_wvar_iapf = []
all_wvar_oapf = []

all_nunique_bpf = []
all_nunique_apf = []
all_nunique_iapf = []
all_nunique_oapf = []

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


	mean_bpf, covs_bpf, ess_bpf, n_unique_bpf, w_vars_bpf, liks_bpf, joint_liks_bpf = bpf.filter(observations)
	mean_apf, covs_apf, ess_apf, n_unique_apf, w_vars_apf, liks_apf, joint_liks_apf = apf.filter(observations)
	mean_iapf, covs_iapf, ess_iapf, n_unique_iapf, w_vars_iapf, liks_iapf, joint_liks_iapf  = iapf.filter(observations)
	mean_oapf, covs_oapf, ess_oapf, n_unique_oapf, w_vars_oapf, liks_oapf, joint_liks_oapf  = oapf.filter(observations)


	all_ess_bpf.append(ess_bpf)
	all_ess_apf.append(ess_apf)
	all_ess_iapf.append(ess_iapf)
	all_ess_oapf.append(ess_oapf)

	all_wvar_bpf.append(w_vars_bpf)
	all_wvar_apf.append(w_vars_apf)
	all_wvar_iapf.append(w_vars_iapf)
	all_wvar_oapf.append(w_vars_oapf)

	all_nunique_bpf.append(n_unique_bpf)
	all_nunique_apf.append(n_unique_apf)
	all_nunique_iapf.append(n_unique_iapf)
	all_nunique_oapf.append(n_unique_oapf)



res_liks = np.vstack([
	all_joint_logliks_bpf,
	all_joint_logliks_apf,
	all_joint_logliks_iapf,
	all_joint_logliks_oapf
])


res_ess = np.vstack([
	all_ess_bpf,
	all_ess_apf,
	all_ess_iapf,
	all_ess_oapf
])

jointliks_bpf = res_liks[:100, :]
jointliks_apf = res_liks[100:200, :]
jointliks_iapf = res_liks[200:300, :]
jointliks_oapf = res_liks[300:400, :]

ess_bpf = res_ess[:100, :]
ess_apf = res_ess[100:200, :]
ess_iapf = res_ess[200:300, :]
ess_oapf = res_ess[300:400, :]



print(np.average(jointliks_bpf))
print(np.average(jointliks_apf))
print(np.average(jointliks_iapf))
print(np.average(jointliks_oapf))

sys.exit()

# res_wvars = np.vstack([
# 	all_wvar_bpf,
# 	all_wvar_apf,
# 	all_wvar_iapf,
# 	all_wvar_oapf
# ])
#
# res_nunique = np.vstack([
# 	all_nunique_bpf,
# 	all_nunique_apf,
# 	all_nunique_iapf,
# 	all_nunique_oapf
# ])


# np.savetxt('results/stochvol/results_stochvol_ess_'+str(n_particle)+'_particles-dim'+str(dim)+'.out', res_ess, delimiter=',')
# np.savetxt('results/stochvol/results_stochvol_wvar_'+str(n_particle)+'_particles-dim'+str(dim)+'.out', res_wvars, delimiter=',')
# np.savetxt('results/stochvol/results_stochvol_nunique_'+str(n_particle)+'_particles-dim'+str(dim)+'.out', res_nunique, delimiter=',')



sys.exit()
# Plotting

ax[0][0].plot(w_vars_bpf, 'b', label='bpf')
ax[0][0].plot(w_vars_apf, 'y', label='apf')
ax[0][0].plot(w_vars_iapf, 'c', label='iapf')
ax[0][0].plot(w_vars_oapf, 'm', label='oapf')
ax[0][0].set(xlabel="timestep",ylabel="w_var")
ax[0][0].set_ylim( (0., np.max( [np.max(w_vars_apf), np.max(w_vars_iapf), np.max(w_vars_oapf)] ) + 1. ))


ax[0][1].plot(ess_bpf, 'b', label='bpf')
ax[0][1].plot(ess_apf, 'y', label='apf')
ax[0][1].plot(ess_iapf, 'c', label='iapf')
ax[0][1].plot(ess_oapf, 'm', label='oapf')
ax[0][1].set(xlabel="timestep",ylabel="ess")
plt.legend()

# plt.savefig('5svm.pdf', bbox_inches='tight')

plt.show()

# i = 0
# plt.plot(states[:,i],'r', label='true_state')
# plt.plot(mean_bpf[:,i],'b', label='mean_bpf')
# plt.plot(mean_oapf[:,i],'m', label='mean_oapf')
# plt.fill_between(np.arange(len(mean_oapf[:,i])), mean_oapf[:,i] - np.sqrt(covs_oapf[:,i,i]), mean_oapf[:,i] + np.sqrt(covs_oapf[:,i,i]), edgecolor=(1 , 0.2, 0.8, 0.99) , facecolor=(1, 0.2, 0.8, 0.3), label="std_oapf", linewidth=1.5)
# plt.fill_between(np.arange(len(mean_bpf[:,i])), mean_bpf[:,i] - np.sqrt(covs_bpf[:,i,i]), mean_bpf[:,i] + np.sqrt(covs_bpf[:,i,i]), edgecolor=(0 , 0, 1, 0.99) , facecolor=(0, 0, 1, 0.3), label="std_bpf", linewidth=1)
# plt.plot(mean_iapf[:,i], '2--',color='c',label='mean_iapf')
# plt.fill_between(np.arange(len(mean_iapf[:,i])), mean_iapf[:,i] - np.sqrt(covs_iapf[:,i,i]), mean_iapf[:,i] + np.sqrt(covs_iapf[:,i,i]), edgecolor=(0.2 , 0.8, 0.8, 0.9) , facecolor=(0.2, 0.8, 0.8, 0.3), label="std_iapf")
# plt.plot(mean_apf[:,i],'y',label='mean_apf')
# plt.fill_between(np.arange(len(mean_apf[:,i])), mean_apf[:,i] - np.sqrt(covs_apf[:,i,i]), mean_apf[:,i] + np.sqrt(covs_apf[:,i,i]), edgecolor=(1, 1, .4, 0.99) , facecolor=(1, 1, .4, 0.3), label="std_apf", linewidth=1.5)
# plt.legend()
# plt.show()
