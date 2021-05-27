import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import *

def create_res(a):
    return (np.average(a, axis=1), np.sqrt(np.var(a, axis=1, ddof=1)) / np.sqrt(a.shape[1]))


set_plotting()

N = 100

res = np.loadtxt('results/lingauss/means/moreparticles_results_lingauss_withoptimal_reduced5_particles-dim10-trvar5.0-obsvar2.5.out',
                 delimiter=',')

res_bpf = res[:11, :]
res_apf = res[11:22, :]
res_iapf = res[22:33, :]
res_oapf = res[33:44, :]
res_faapf = res[44:55, :]
res_approx_kf = res[55:66, :]

res_bpf = create_res(res_bpf)
res_apf = create_res(res_apf)
res_iapf = create_res(res_iapf)
res_oapf = create_res(res_oapf)
res_faapf = create_res(res_faapf)
res_approx_kf = create_res(res_approx_kf)


particles = [10,20,40,60,80,100,200,400,600,800,1000]


plt.plot(particles, res_bpf[0], color='b', marker='o', label='bpf', linewidth=0.2, markersize=5)

plt.plot(particles, res_apf[0], color=(1, 0.4, 0),marker='o', label='apf', linewidth=0.2, markersize=5)

plt.plot(particles, res_iapf[0], color='c', marker='o', label='iapf', linewidth=0.2, markersize=5)

plt.plot(particles, res_oapf[0], color='m', marker='o', label='oapf', linewidth=0.2, markersize=5)

plt.plot(particles, res_faapf[0], color='g', marker='o', label='fa-apf', linewidth=0.2, markersize=5)

plt.plot(particles, res_approx_kf[0], color='r', marker='o', label='true-posterior-samples', linewidth=0.2, markersize=5)

plt.fill_between(particles,
                 np.asarray(res_bpf[0]) - 2.5 *np.asarray(res_bpf[1]),
                 np.asarray(res_bpf[0]) + 2.5 *np.asarray(res_bpf[1]),
                 edgecolor=(0, 0, 1, 0.99), facecolor=(0, 0, 1, 0.4))

plt.fill_between(particles,
                 res_apf[0] - 2.5 * res_apf[1],
                 res_apf[0] + 2.5 * res_apf[1],
                 edgecolor=(1, 0.4, 0), facecolor=(1, 0.4, 0, 0.4))

plt.fill_between(particles,
                 res_iapf[0] - 2.5 * res_iapf[1],
                 res_iapf[0] + 2.5 * res_iapf[1],
                 edgecolor=(0.2 , 0.8, 0.8, 0.9) , facecolor=(0.2, 0.8, 0.8, 0.4))

plt.fill_between(particles,
                 res_oapf[0] - 2.5 * res_oapf[1],
                 res_oapf[0] + 2.5 * res_oapf[1],
                 edgecolor=(1 , 0.2, 0.8, 0.99) , facecolor=(1, 0.2, 0.8, 0.4))

plt.fill_between(particles,
                 res_faapf[0] - 2.5 * res_faapf[1],
                 res_faapf[0] + 2.5 * res_faapf[1],
                 edgecolor=(0.5, 0.6, 0.2, 0.99), facecolor=(0.5, 0.6, 0.2, 0.4))

plt.fill_between(particles,
                 res_approx_kf[0] - 2.5 * res_approx_kf[1],
                 res_approx_kf[0] + 2.5 * res_approx_kf[1],
                 edgecolor=(1, 0, 0, 0.99), facecolor=(1, 0, 0, 0.4))
plt.ylabel('Normalized MSE (posterior mean)')
plt.xlabel('Number of particles')
# plt.xscale('log')
# plt.ylim((0.,0.1))
plt.yscale('log')
plt.tight_layout()
plt.legend()
# plt.show()
plt.savefig('imgs/comparison_means_dim10-trvar5.0-obsvar2.5.pdf',bbox_inches='tight')