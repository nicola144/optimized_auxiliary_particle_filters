
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import *

set_plotting()

res_ess = np.loadtxt('results/stochvol/results_stochvol_ess_1000_particles-dim10.out',  delimiter=',')
res_wvar = np.loadtxt('results/stochvol/results_stochvol_wvar_1000_particles-dim10.out',  delimiter=',')
res_nunique = np.loadtxt('results/stochvol/results_stochvol_nunique_1000_particles-dim10.out',  delimiter=',')

ess_bpf = res_ess[:100, :]
ess_apf = res_ess[100:200, :]
ess_iapf = res_ess[200:300, :]
ess_oapf = res_ess[300:400, :]

wvar_bpf = res_wvar[:100, :]
wvar_apf = res_wvar[100:200, :]
wvar_iapf = res_wvar[200:300, :]
wvar_oapf = res_wvar[300:400, :]

wvar_bpf_mean = np.average(wvar_bpf, axis=1)
wvar_apf_mean = np.average(wvar_apf, axis=1)
wvar_iapf_mean = np.average(wvar_iapf, axis=1)
wvar_oapf_mean = np.average(wvar_oapf, axis=1)

ess_bpf_mean = np.average(ess_bpf, axis=1)
ess_apf_mean = np.average(ess_apf, axis=1)
ess_iapf_mean = np.average(ess_iapf, axis=1)
ess_oapf_mean = np.average(ess_oapf, axis=1)

ess_bpf_var = np.var(ess_bpf, ddof=1, axis=1)
ess_apf_var = np.var(ess_apf, ddof=1, axis=1)
ess_iapf_var = np.var(ess_iapf,  ddof=1,axis=1)
ess_oapf_var = np.var(ess_oapf, ddof=1, axis=1)





# plt.plot(wvar_bpf_mean, color=(0, 0, 1), alpha=0.8)
# plt.plot(wvar_apf_mean, color=(0, 230./255., 115/255.), alpha=0.8)
# plt.plot(wvar_iapf_mean, color=(0.2, 0.8, 0.8), alpha=0.8)
# plt.plot(wvar_oapf_mean, color=(1, 0.2, 0.8), alpha=0.8)
# plt.ylim((0.,100.))
# plt.ylim( (0., np.max( [np.max(wvar_apf_mean), np.max(wvar_iapf_mean), np.max(wvar_oapf_mean)] ) + 1. ))

plt.plot(ess_bpf_mean, color=(0, 0, 1), alpha=0.8)
plt.plot(ess_apf_mean, color=(0.8, 0.4, 0), alpha=0.8)
plt.plot(ess_iapf_mean, color=(0.2, 0.8, 0.8), alpha=0.8)
plt.plot(ess_oapf_mean, color=(1, 0.2, 0.8), alpha=0.8)

N = np.sqrt(ess_bpf_mean.shape[0])
plt.fill_between(np.arange(len(ess_bpf_mean)), ess_bpf_mean - np.sqrt(ess_bpf_var)/np.sqrt(N), ess_bpf_mean + np.sqrt(ess_bpf_var)/np.sqrt(N),  edgecolor=(0 , 0, 1, 0.99) , facecolor=(0, 0, 1, 0.3), label="std_bpf", linewidth=1)
plt.fill_between(np.arange(len(ess_apf_mean)), ess_apf_mean - np.sqrt(ess_apf_var)/np.sqrt(N), ess_apf_mean + np.sqrt(ess_apf_var)/np.sqrt(N),  edgecolor=(0.8, 0.4, 0., 0.99) , facecolor=(0.8, 0.4, 0., 0.3), label="std_apf", linewidth=1)
plt.fill_between(np.arange(len(ess_iapf_mean)), ess_iapf_mean - np.sqrt(ess_iapf_var)/np.sqrt(N), ess_iapf_mean + np.sqrt(ess_iapf_var)/np.sqrt(N),  edgecolor=(0.2, 0.8, 0.8, 0.99) , facecolor=(0.2, 0.8, 0.8, 0.3), label="std_iapf", linewidth=1)
plt.fill_between(np.arange(len(ess_oapf_mean)), ess_oapf_mean - np.sqrt(ess_oapf_var)/np.sqrt(N), ess_oapf_mean + np.sqrt(ess_oapf_var)/np.sqrt(N),  edgecolor=(1, 0.2, 0.8,0.99), facecolor=(1, 0.2, 0.8,0.3), label="std_oapf", linewidth=1)
plt.legend()
plt.title("Stoch. Volatility: ESS vs Time, 100 Monte Carlo simulations")
plt.xlabel("Timestep")
plt.ylabel("ESS")
plt.show()
# plt.savefig("imgs/ess-comparison1.pdf", bbox_inches='tight')
sys.exit()

# nunique_bpf = res_nunique[:100, :]
# nunique_apf = res_nunique[100:200, :]
# nunique_iapf = res_nunique[200:300, :]
# nunique_oapf = res_nunique[300:400, :]


for col in wvar_bpf.T:
	plt.plot(col, color=(0, 0, 1), alpha=0.2)
for col in wvar_apf.T:
	plt.plot(col, color=(0, 230./255., 115/255.), alpha=0.2)
for col in wvar_iapf.T:
	plt.plot(col, color=(0.2, 0.8, 0.8), alpha=0.2)
for col in wvar_oapf.T:
	plt.plot(col,color=(1, 0.2, 0.8),alpha=0.2)
# plt.ylim( (0., np.max( [np.max(wvar_apf), np.max(wvar_iapf), np.max(wvar_oapf)] ) + 1. ))
plt.ylim((0.,100.))
plt.show()
sys.exit()


plt.plot(w_vars_bpf, 'b', label='bpf')
plt.plot(w_vars_apf, 'y', label='apf')
plt.plot(w_vars_iapf, 'c', label='iapf')
plt.plot(w_vars_oapf, 'm', label='oapf')
plt.xlabel("timestep")
plt.ylabel("w_var")
plt.ylim( (0., np.max( [np.max(w_vars_apf), np.max(w_vars_iapf), np.max(w_vars_oapf)] ) + 1. ))


plt.plot(ess_bpf, 'b', label='bpf')
plt.plot(ess_apf, 'y', label='apf')
plt.plot(ess_iapf, 'c', label='iapf')
plt.plot(ess_oapf, 'm', label='oapf')
plt.xlabel("timestep")
plt.ylabel("ess")
plt.legend()



plt.show()
# plt.savefig('results/boxplot-'+ str(n_particle)+ '-reduced-res.pdf', bbox_inches='tight')

