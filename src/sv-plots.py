
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import *

set_plotting()



res = np.loadtxt('/Users/nicolabranchini/PycharmProjects/auxiliary-particle-filters/results/stochvol/joint_logliks/moreparticles_results_stochvol_reduced5_jointliks-seq-dim10varcov-2.5-varphi_corr_oneovertwo.out',
                 delimiter=',')


liks_bpf = res[:100, :]
liks_apf = res[100:200, :]
liks_iapf = res[200:300, :]
liks_oapf = res[300:400, :]

N = liks_bpf.shape[0]

# lorenz variances
#
# 80780.70199955124
# 80838.82231599772
# 79422.8795177361
# 78984.66638196257


liks_bpf_mean = np.average(liks_bpf, axis=1)
liks_apf_mean = np.average(liks_apf, axis=1)
liks_iapf_mean = np.average(liks_iapf, axis=1)
liks_oapf_mean = np.average(liks_oapf, axis=1)


plt.plot(liks_bpf_mean, color=(0, 0, 1), alpha=0.8)
plt.plot(liks_apf_mean, color=(0.8, 0.4, 0), alpha=0.8)
plt.plot(liks_iapf_mean, color=(0.2, 0.8, 0.8), alpha=0.8)
plt.plot(liks_oapf_mean, color=(1, 0.2, 0.8), alpha=0.8)
plt.tight_layout()
plt.xscale('log')
plt.show()





