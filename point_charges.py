import numpy as np
import matplotlib.pyplot as plt

from electromagnetic import FreeFieldVectorPotential
import reduced_parameter as red

"""
Testing electromagnetic.py
"""

k_vector = np.array([[0,0,1]]) / red.c
amplitude = np.array(
        [np.random.uniform(size = 2) * 100 + np.random.uniform(size = 2) * 100j]
        )

Afield = FreeFieldVectorPotential(
        k_vector = k_vector, amplitude = amplitude,
        V = 1.0, constant_c = red.c
        )

#simple point charge
r = np.array([[1.0,1.0,1.0]])
v = np.array([[1.0,0.0,0.0]]) * 10
q = np.array([10 * np.eye(3)])
m = 1

t = 0
h = 1e-4

def current(r_dot, q):
    return np.einsum("nii,ni->ni",q,r_dot)

def EM_force(t, r, r_dot, q, A):

    dAdt = A.time_diff(t,r,r,current(r_dot,q))
    gradA = A.gradient(t,r)

    force1 = np.einsum("nlj,nl->nj",q, r_dot)
    force1 = np.einsum("nj,nji->ni",force1,gradA)

    force2 = np.einsum("nj,nji->ni",dAdt, q)

    force3 = np.einsum("njl,nl->nj",gradA,r_dot)
    force3 = np.einsum("nj,nij->ni",force3,q)

    force1 /= A.constant_c
    force2 /= A.constant_c
    force3 /= A.constant_c
    force = force1 - force2 - force3

    return force

def kinetic_energy(r_dot):
    k = 0.5 * r_dot.T @ r_dot
    return np.sum(k)

Hmat = kinetic_energy(v)
Hrad =  Afield.hamiltonian(True)

Hmat_list = [Hmat]
Hrad_list = [Hrad]
energy = [Hmat + Hrad]
time = [0]

#first iteration w/ Euler integration (and trapezoidal rule)

for i in range(10000):
    force = EM_force(t, r, v, q, Afield)
    v_half = v + force * h / 2
    r_new = r + v_half * h

    new_force = EM_force(t + h, r_new, v_half, q, Afield)
    v_new = v_half + new_force * h / 2

    C_1 = Afield.dot_amplitude(t,r,current(v,q))
    C_2 = Afield.dot_amplitude(t+h,r_new,current(v_new,q))
    C_new = Afield.C + h * (C_1 + C_2) / 2

    t += h
    r = r_new
    v = v_new

    Afield.update_amplitude(C_new)

    Hmat = kinetic_energy(v)
    Hrad =  Afield.hamiltonian(True)

    energy.append(Hmat + Hrad)
    Hmat_list.append(Hmat)
    Hrad_list.append(Hrad)
    time.append(t)

fig,ax = plt.subplots()
ax.plot(time,energy)

fig.savefig("test.jpeg")

fig,ax = plt.subplots()
ax.plot(time,Hmat_list)

fig.savefig("test2.jpeg")

fig,ax = plt.subplots()
ax.plot(time,Hrad_list)

fig.savefig("test3.jpeg")






