"""
================
Lorenz Attractor
================

This is an example of plotting Edward Lorenz's 1963 `"Deterministic Nonperiodic
Flow"`_ in a 3-dimensional space using mplot3d.

.. _"Deterministic Nonperiodic Flow":
   http://journals.ametsoc.org/doi/abs/10.1175/1520-0469%281963%29020%3C0130%3ADNF%3E2.0.CO%3B2

.. note::
   Because this is a simple non-linear ODE, it would be more easily done using
   SciPy's ODE solver, but this approach depends only upon NumPy.
"""
from utils import *
from particles import *
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# data generation
dim = 3
timesteps = 1000
n_particle = 1000
dt = 0.01

s=10
r=28
b=2.667

set_plotting()
seeds = np.loadtxt('seeds.out').astype('int64')

constant_mean = np.zeros(dim, )
initial_cov = np.eye(dim)


def prior_sample(random_state, size=1):
    res = random_state.multivariate_normal(mean=constant_mean,cov=initial_cov,size=size)
    return res.squeeze()

def lorenz(x, y, z, s, r, b):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


for seed in tqdm(seeds):

    random_state = np.random.RandomState(seed)

    observations = []

    # Need one more for the initial values
    xs = np.empty(timesteps + 1)
    ys = np.empty(timesteps + 1)
    zs = np.empty(timesteps + 1)

    # Set initial values
    xs[0], ys[0], zs[0] = tuple(prior_sample(random_state))

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(timesteps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
        xs[i + 1] = xs[i] + (x_dot * dt) + random_state.randn()
        ys[i + 1] = ys[i] + (y_dot * dt) + random_state.randn()
        zs[i + 1] = zs[i] + (z_dot * dt) + random_state.randn()

        observations.append(xs[i + 1] + random_state.normal(0,1))

    observations = np.asarray(observations)

    states = np.vstack([xs,ys,zs]).T

    bpf = LorenzBPF(init_particle=prior_sample(random_state,size=n_particle),
                      random_state=random_state,
                      s=s,
                      r=r,
                      b=b,
                      delta=dt,
                      transition_cov=np.eye(3),
                      observation_var = np.eye(1))

    apf = LorenzAPF(init_particle=prior_sample(random_state,size=n_particle),
                      random_state=random_state,
                      s=s,
                      r=r,
                      b=b,
                      delta=dt,
                      transition_cov=np.eye(3),
                      observation_var = np.eye(1))

    iapf = LorenzIAPF(init_particle=prior_sample(random_state,size=n_particle),
                      random_state=random_state,
                      s=s,
                      r=r,
                      b=b,
                      delta=dt,
                      transition_cov=np.eye(3),
                      observation_var = np.eye(1))

    oapf = LorenzOAPF(init_particle=prior_sample(random_state,size=n_particle),
                      random_state=random_state,
                      s=s,
                      r=r,
                      b=b,
                      delta=dt,
                      transition_cov=np.eye(3),
                      observation_var = np.eye(1))


    mean_bpf, covs_bpf,  ess_bpf, n_unique_bpf, w_vars_bpf, liks_bpf, joint_liks_bpf = bpf.filter(observations)
    mean_apf, covs_apf,  ess_apf, n_unique_apf, w_vars_apf, liks_apf, joint_liks_apf = apf.filter(observations)
    mean_iapf, covs_iapf,  ess_iapf, n_unique_iapf, w_vars_iapf, liks_iapf, joint_liks_iapf = iapf.filter(observations)
    mean_oapf, covs_npf,  ess_npf, n_unique_npf, w_vars_npf, liks_oapf, joint_liks_oapf = oapf.filter(observations)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(mean_bpf[:,0], mean_bpf[:,1], mean_bpf[:,2], 'b', lw=0.5)
    ax.plot(mean_apf[:,0], mean_apf[:,1], mean_apf[:,2], 'y', lw=0.5)
    ax.plot(mean_iapf[:,0], mean_iapf[:,1], mean_iapf[:,2], 'c', lw=0.5)
    ax.plot(mean_oapf[:,0], mean_oapf[:,1], mean_oapf[:,2], 'm', lw=0.5)
    ax.plot(states[:,0], states[:,1], states[:,2], 'r', lw=0.5)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plt.show()

    sys.exit()








# Plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.plot(xs, ys, zs, lw=0.5)
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_title("Lorenz Attractor")

# plt.show()
