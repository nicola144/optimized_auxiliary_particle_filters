
import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import *

set_plotting()

res = np.loadtxt('results/results_lingauss_100_reduced_particles-dim2-equal-offset.out',  delimiter=',')

n_monte_carlo = 100000

averages = [[] for algorithm in range(res.shape[0])]

for algorithm in range(res.shape[0]):
    averages[algorithm].append(np.mean(res[algorithm]))

    for i in range(n_monte_carlo):
        averages[algorithm].append(np.mean(np.random.choice(a=res[algorithm],replace=True)))

averages = np.asarray(averages)

# d = plt.boxplot([averages[0],averages[1],averages[2],averages[3]], labels=['bpf','apf','iapf','oapf'], whis=1.5)
d = plt.boxplot([averages[2],averages[3]], labels=['iapf','oapf'], whis=1.5)


# plt.boxplot([res[0],res[1], res[2],res[3]], labels=['bpf','apf', 'iapf','oapf'], bootstrap=5000)

# d = plt.boxplot([res[2],res[3]], labels=['iapf','oapf'], bootstrap=100, notch=True)

print(d['medians'][0].get_ydata(orig=True)[0])
print(d['medians'][1].get_ydata(orig=True)[0])

plt.axhline(y=d['medians'][0].get_ydata(orig=True)[0], color='r', linestyle='-')
plt.axhline(y=d['medians'][1].get_ydata(orig=True)[0], color='b', linestyle='-')


plt.show()
# plt.savefig('results/boxplot-'+ str(n_particle)+ '-reduced-res.pdf', bbox_inches='tight')

sys.exit()
