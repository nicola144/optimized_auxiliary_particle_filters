
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import *

set_plotting()


# plt.yscale('log')


res_liks = np.loadtxt('results/lingauss/joint_logliks/results_lingauss_1000_reduced5_particles-dim10-trvar5.0-obsvar2.5.out',  delimiter=',')

N = res_liks.shape[1]
print(np.average(res_liks, axis=1))
print(np.sqrt(np.var(res_liks,ddof=1,axis=1)) / np.sqrt(N) )





# n_monte_carlo = 100000

# averages = [[] for algorithm in range(res.shape[0])]
#
# for algorithm in range(res.shape[0]):
#     averages[algorithm].append(np.mean(res[algorithm]))
#
#     for i in range(n_monte_carlo):
#         averages[algorithm].append(np.mean(np.random.choice(a=res[algorithm],replace=True)))
#
# averages = np.asarray(averages)
#
# d = plt.boxplot([averages[0],averages[1],averages[2],averages[3]], labels=['bpf','apf','iapf','oapf'], whis=1.5)
# # d = plt.boxplot([averages[2],averages[3]], labels=['iapf','oapf'], whis=1.5)
#

d = plt.boxplot([res_liks[0],res_liks[1], res_liks[2],res_liks[3]], labels=['BPF','APF', 'IAPF','OAPF'], bootstrap=10000, notch=True)

# d = plt.boxplot([res[2],res[3]], labels=['iapf','oapf'], bootstrap=100, notch=True)

print(d['medians'][0].get_ydata(orig=True)[0])
print(d['medians'][1].get_ydata(orig=True)[0])
print(d['medians'][2].get_ydata(orig=True)[0])
print(d['medians'][3].get_ydata(orig=True)[0])

# plt.axhline(y=d['medians'][0].get_ydata(orig=True)[0], color='r', linestyle='-')
# plt.axhline(y=d['medians'][1].get_ydata(orig=True)[0], color='b', linestyle='-')


# plt.show()






plt.savefig('results/boxplot-FINAL.pdf', bbox_inches='tight')

# 0.0031599361015733915
# 0.0027931818448951977
# 0.0020850581157131763
# 0.0016579833609904312

sys.exit()
