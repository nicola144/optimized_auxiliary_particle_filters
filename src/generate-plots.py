
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import *

set_plotting()

# res_liks = np.loadtxt('results/lorenz/joint_logliks/red10oapf-results_lorenz_jointliks_1000_particles-dim3-timest-100.out',  delimiter=',')

# res_ess = np.loadtxt('results/lorenz/ess/RED20-results_lorenz_ess_100_particles-dim3-timest-100.out',  delimiter=',')
# res_ess = np.loadtxt('results/lorenz/joint_logliks/RED50-results_lorenz_jointliks_100_particles-dim3-timest-100.out',  delimiter=',')
# res_ess = np.loadtxt('results/lorenz/ess/PAPER-dt0.008-results_lorenz_ess_100_particles-dim3-timest-1000.out',  delimiter=',')
# res_liks = np.loadtxt('results/stochvol/joint_logliks/notred-results_stochvol_jointliks_100_particles-dim5.out',  delimiter=',')


res_ess = np.loadtxt('results/stochvol/ess/NOTREDPAPER-results_stochvol_ess_1000_particles-dim10.out',  delimiter=',')
# res_liks = np.loadtxt('results/stochvol/joint_logliks/results_stochvol_jointliks_1000_particles-dim10.out',  delimiter=',')

ess_bpf = res_ess[:100, :]
ess_apf = res_ess[100:200, :]
ess_iapf = res_ess[200:300, :]
ess_oapf = res_ess[300:400, :]



# liks_bpf = res_liks[:100, :]
# liks_apf = res_liks[100:200, :]
# liks_iapf = res_liks[200:300, :]
# liks_oapf = res_liks[200:300, :]

N = ess_bpf.shape[0]

# lorenz variances
#
# 80780.70199955124
# 80838.82231599772
# 79422.8795177361
# 78984.66638196257


ess_bpf_mean = np.average(ess_bpf, axis=1)
ess_apf_mean = np.average(ess_apf, axis=1)
ess_iapf_mean = np.average(ess_iapf, axis=1)
ess_oapf_mean = np.average(ess_oapf, axis=1)

print(np.average(ess_bpf_mean))
print(np.average(ess_apf_mean))
print(np.average(ess_iapf_mean))
print(np.average(ess_oapf_mean))


print(np.sqrt(np.var(np.average(ess_bpf,axis=0), ddof=1)) / np.sqrt(N))
print(np.sqrt(np.var(np.average(ess_apf,axis=0), ddof=1)) / np.sqrt(N))
print(np.sqrt(np.var(np.average(ess_iapf,axis=0), ddof=1)) / np.sqrt(N))
print(np.sqrt(np.var(np.average(ess_oapf,axis=0), ddof=1)) / np.sqrt(N))
# sys.exit()

ess_bpf_var = np.var(ess_bpf, ddof=1, axis=1)
ess_apf_var = np.var(ess_apf, ddof=1, axis=1)
ess_iapf_var = np.var(ess_iapf,  ddof=1,axis=1)
ess_oapf_var = np.var(ess_oapf, ddof=1, axis=1)

# liks_bpf_mean = np.average(liks_bpf, axis=1)
# liks_apf_mean = np.average(liks_apf, axis=1)
# liks_iapf_mean = np.average(liks_iapf, axis=1)
# liks_oapf_mean = np.average(liks_oapf, axis=1)
# #
# liks_bpf_var = np.var(liks_bpf, ddof=1, axis=1) / np.sqrt(N)
# liks_apf_var = np.var(liks_apf, ddof=1, axis=1) / np.sqrt(N)
# liks_iapf_var = np.var(liks_iapf,  ddof=1,axis=1) / np.sqrt(N)
# liks_oapf_var = np.var(liks_oapf, ddof=1, axis=1) / np.sqrt(N)
#
# print(np.average(liks_bpf_var) )
# print(np.average(liks_apf_var) )
# print(np.average(liks_iapf_var) )
# print(np.average(liks_oapf_var) )
#

plt.plot(ess_bpf_mean, color=(0, 0, 1), alpha=0.8)
plt.plot(ess_apf_mean, color=(0.8, 0.4, 0), alpha=0.8)
plt.plot(ess_iapf_mean, color=(0.2, 0.8, 0.8), alpha=0.8)
plt.plot(ess_oapf_mean, color=(1, 0.2, 0.8), alpha=0.8)

plt.fill_between(np.arange(len(ess_bpf_mean)), ess_bpf_mean - np.sqrt(ess_bpf_var)/np.sqrt(N), ess_bpf_mean + np.sqrt(ess_bpf_var)/np.sqrt(N),  edgecolor=(0 , 0, 1, 0.99) , facecolor=(0, 0, 1, 0.3), label="std_bpf", linewidth=1)
plt.fill_between(np.arange(len(ess_apf_mean)), ess_apf_mean - np.sqrt(ess_apf_var)/np.sqrt(N), ess_apf_mean + np.sqrt(ess_apf_var)/np.sqrt(N),  edgecolor=(0.8, 0.4, 0., 0.99) , facecolor=(0.8, 0.4, 0., 0.3), label="std_apf", linewidth=1)
plt.fill_between(np.arange(len(ess_iapf_mean)), ess_iapf_mean - np.sqrt(ess_iapf_var)/np.sqrt(N), ess_iapf_mean + np.sqrt(ess_iapf_var)/np.sqrt(N),  edgecolor=(0.2, 0.8, 0.8, 0.99) , facecolor=(0.2, 0.8, 0.8, 0.3), label="std_iapf", linewidth=1)
plt.fill_between(np.arange(len(ess_oapf_mean)), ess_oapf_mean - np.sqrt(ess_oapf_var)/np.sqrt(N), ess_oapf_mean + np.sqrt(ess_oapf_var)/np.sqrt(N),  edgecolor=(1, 0.2, 0.8,0.99), facecolor=(1, 0.2, 0.8,0.3), label="std_oapf", linewidth=1)
plt.legend()
plt.title("Stochastic Volatility: ESS vs Time")
plt.xlabel("Timestep")
plt.ylabel("ESS")
# plt.show()
plt.savefig("imgs/ess-comparison-10dim-FINAL.pdf", bbox_inches='tight')
sys.exit()


plt.plot(liks_bpf_mean, color=(0, 0, 1), alpha=0.8)
plt.plot(liks_apf_mean, color=(0.8, 0.4, 0), alpha=0.8)
# plt.plot(liks_iapf_mean, color=(0.2, 0.8, 0.8), alpha=0.8)
plt.plot(liks_oapf_mean, color=(1, 0.2, 0.8), alpha=0.8)

N = np.sqrt(liks_bpf_mean.shape[0])
plt.fill_between(np.arange(len(liks_bpf_mean)), liks_bpf_mean - np.sqrt(liks_bpf_var)/np.sqrt(N), liks_bpf_mean + np.sqrt(liks_bpf_var)/np.sqrt(N),  edgecolor=(0 , 0, 1, 0.99) , facecolor=(0, 0, 1, 0.3), label="std_bpf", linewidth=1)
plt.fill_between(np.arange(len(liks_apf_mean)), liks_apf_mean - np.sqrt(liks_apf_var)/np.sqrt(N), liks_apf_mean + np.sqrt(liks_apf_var)/np.sqrt(N),  edgecolor=(0.8, 0.4, 0., 0.99) , facecolor=(0.8, 0.4, 0., 0.3), label="std_apf", linewidth=1)
# plt.fill_between(np.arange(len(liks_iapf_mean)), liks_iapf_mean - np.sqrt(liks_iapf_var)/np.sqrt(N), liks_iapf_mean + np.sqrt(liks_iapf_var)/np.sqrt(N),  edgecolor=(0.2, 0.8, 0.8, 0.99) , facecolor=(0.2, 0.8, 0.8, 0.3), label="std_iapf", linewidth=1)
plt.fill_between(np.arange(len(liks_oapf_mean)), liks_oapf_mean - np.sqrt(liks_oapf_var)/np.sqrt(N), liks_oapf_mean + np.sqrt(liks_oapf_var)/np.sqrt(N),  edgecolor=(1, 0.2, 0.8,0.99), facecolor=(1, 0.2, 0.8,0.3), label="std_oapf", linewidth=1)
plt.legend()
plt.title("Stoch. Volatility: Liks vs Time, 100 Monte Carlo simulations")
plt.xlabel("Timestep")
plt.ylabel("LIKS")
# plt.show()
plt.savefig("imgs/ess-comparison-FINAL.pdf", bbox_inches='tight')


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



# plt.show()
# plt.savefig('results/boxplot-'+ str(n_particle)+ '-reduced-res.pdf', bbox_inches='tight')

