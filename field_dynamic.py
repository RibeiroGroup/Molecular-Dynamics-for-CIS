from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import constants

k = 2 * np.pi / (700e-9 / constants.a0)
k_vec = np.array([k,0,0])
omega = constants.c * k

h = 1e-2
C = np.random.rand(2)
C1_list = [C[0]]
C2_list = [C[1]]
energy_list = [np.real(C @ np.conjugate(C).T)]

# C_dot = - i omega C
def dot_C(C):
    return -1j * omega * C 

n_steps = 20000
for i in range(n_steps):
    k1 = dot_C(C)
    k2 = dot_C(C + h*k1/2)
    k3 = dot_C(C + h*k2/2)
    k4 = dot_C(C + h*k3)

    C = C + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    C1_list.append(deepcopy(C[0]))
    C2_list.append(deepcopy(C[1]))
    energy_list.append(np.real(C @ np.conjugate(C).T))

C1_list = np.array(C1_list)
C2_list = np.array(C2_list)

fig, ax = plt.subplots()

ax.plot(np.real(C1_list), np.imag(C1_list))
ax.plot(np.real(C2_list), np.imag(C2_list))

fig.savefig("result_plot/free_field_dyanmic.jpeg",dpi = 600)

fig, ax = plt.subplots()

ax.plot(np.arange(0,n_steps+1), energy_list)
ax.set_ylim(-1,1)

fig.savefig("result_plot/free_field_energy.jpeg",dpi = 600)

