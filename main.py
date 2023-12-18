from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import vector_potential
from num_pde import RK4
from test_cases import A_test_case

np.random.seed(2023)

# defining parameters for the EM field

k_vec = np.array([[1e-1,0,0]]) * np.random.rand()
epsilon = np.array([
    [0, 1, 0], [0, 0, 1]
    ])
C0 = (np.random.rand(2) + 1j *  np.random.rand(2)) * 1e2

c = 1.37036e2
k = np.sqrt(k_vec @ k_vec.T)
omega = c * k

A = vector_potential(C=C0, k=k_vec, epsilon=epsilon)

# particle with random position and velocity
ra0 = np.random.rand() * np.array([1,0,0]) # np.random.rand(3)
va0 = np.random.rand() * np.array([0,0,0]) #np.random.rand(3)
qa = 1

# choosing time step

h = 0.001

time_step = 500

ra_list = [ra0]
va_list = [va0]
C_list = [A.C]
energy_list = []

def f(ra, va, C):
    jk = A.transverse_project(
        (1/(2*np.pi**1.5)) * np.exp(-1j * k_vec @ ra) * qa * va
        )

    return - 1j * omega * C + \
        np.einsum('ijk,kj->ik', np.tile(jk[:,:,np.newaxis],(1,1,2)), epsilon)

for i in range(time_step):
    #Verlett update
    ra = ra_list[-1]
    va = va_list[-1]

    F = (qa/c) * (va - (qa/c) * A(ra)) @ A.diff_ra(ra)

    if len(ra_list) < 2:
        ra_new = ra + va * h + 0.5 * F * h**2
        va_new = (ra_new - ra) / h
    else:
        ra_1 = ra_list[-2]
        ra_new = 2*ra - ra_1 + F * h**2
        va_new = (ra_new - ra_1) / (2*h)

    #euler/rk4 update

    if len(C_list) < 2:
        C = C_list[-1]
        C_new = C + h*f(ra, va, C)
    else:
        C_1 = C_list[-2]
        va_1 = va_list[-2]

        k1 = f(ra_1, va_1, C_1)
        k2 = f(ra, va , C_1 + h*k1)
        k3 = f(ra, va,  C_1 + h*k2)
        k4 = f(ra_new, va_new, C_1 + 2*h*k3)

        C_new = C_1 + (h/3) * (k1 + 2*k2 + 2*k3 + k4)

    ra_list.append(ra_new)
    va_list.append(va_new)
    C_list.append(C_new)

    energy = 0.5 * (va_new - (qa/c) * A(ra_new)).T @ (va_new - (qa/c) * A(ra_new))
    energy += k**2 * np.sum(np.einsum('ij,ij->i',C_new, np.conjugate(C_new))) / (2*np.pi)
    print(energy)
    energy_list.append(np.real(energy.ravel()[0]))

fig,ax = plt.subplots()
    
ax.scatter(np.arange(0,time_step,1)*h, energy_list)

fig.savefig("energy.jpeg",dpi = 600)

fig,ax = plt.subplots()
    
ra_list = np.array(ra_list)
ax.plot(np.arange(0,time_step+1,1)*h, ra_list[:,0])

fig.savefig("trajectory.jpeg",dpi = 600)

