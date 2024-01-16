import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
from Charge import ChargePoint, ChargeCluster
from simpleForceField import MorsePotential, compute_Morse_force, compute_Hmorse

import constants
from simpleMD import *

np.random.seed(20)
# FIELD SPECS
k_ = 2 * np.pi / (100e-9 / constants.a0)
A = MultiModeField(
    C= (np.random.rand(2) + 1j * np.random.rand(2)),
    k=np.array([k_,0,0]),
    epsilon=np.array([[0,1,0], [0,0,1]])
    )

#PARTICLE SPECS 
alpha = ChargePoint(
        m = 1, q = 1, 
        r = np.random.rand(3),
        v = np.random.rand(3), # np.zeros(3), # 
        )

beta = ChargePoint(
        m = 1, q = -1, 
        r = np.zeros(3), # np.random.rand(3), #
        v = np.random.rand(3)
        )

print("####### Initial field parameters value #######")
C = A.C[0]
print("C = ",C)
k_vec = deepcopy(A.k[0])
print("k = ",k_vec)
epsilon = np.array(A.epsilon[0])
print("epsilon = ",epsilon)
h = 1e-2
print("h = ", h)

print("### Initial charge point parameters value ###")
q = [beta.q]#, alpha.q]
print("q = ",q)
r = np.vstack([beta.r])#, alpha.r])
r = r.reshape(-1,3)
print("r = ",r)
v = np.vstack([beta.v])#, alpha.v])
v = v.reshape(-1,3)
print("v = ",v)

print("############# others #############")

box_dimension = np.array([4]*3)
print("box dimension:", box_dimension)

k_const = None
print("oscillator constant k_const",k_const)

De = 1495 / 4.35975e-18 / 6.023e23
Re = (3.5e-10) / 5.29177e-11
a = 1/ ( (1/3 * 1e-10) / 5.29177e-11)
potential = None # MorsePotential(De=De, Re=Re, a=a)

print("#############################################")

Hem, Hmat = compute_H(
        q=q,r=r,v=v,k_vec=k_vec,C=C,epsilon=epsilon,k_const=k_const,potential=potential)

print("Hamiltonian: {}(field) + {}(matter)".format(Hem,Hmat))

em_H_list = [Hem]
mat_H_list = [Hmat]
H_list = [Hem + Hmat]
steps_list = [0]

trajectory = {"initial":{"q":q,"r":r,"v":v,"k_const":k_const},
        "r":[r], "v":[v]}
hamiltonian = {"em":[Hem], "mat":[Hmat]}

for i in range(int(2e3+1)):
    k1c = dot_C(
        q=q, r=r, v=v, C=C, k_vec=k_vec, epsilon=epsilon)
    k1v = compute_force(
        q=q, r=r, v=v, C=C, k_vec=k_vec, epsilon=epsilon,
        k_const=k_const,potential=potential)
    k1r = v

    k2c = dot_C(
        q=q, r=r+k1r*h/2, v=v+k1v*h/2, C=C+k1c*h/2, k_vec=k_vec, epsilon=epsilon)
    k2v = compute_force(
        q=q, r=r+k1r*h/2, v=v+k1v*h/2, C=C+k1c*h/2, k_vec=k_vec, epsilon=epsilon,
        k_const=k_const,potential=potential)
    k2r = v + k1v*h/2

    k3c = dot_C(
        q=q, r=r+k2r*h/2, v=v+k2v*h/2, C=C+k2c*h/2, k_vec=k_vec, epsilon=epsilon)
    k3v = compute_force(
        q=q, r=r+k2r*h/2, v=v+k2v*h/2, C=C+k2c*h/2, k_vec=k_vec, epsilon=epsilon,
        k_const=k_const,potential=potential)
    k3r = v + k2v*h/2

    k4c = dot_C(
        q=q, r=r+k3r*h, v=v+k3v*h, C=C+k3c*h, k_vec=k_vec, epsilon=epsilon)
    k4v = compute_force(
        q=q, r=r+k3r*h, v=v+k3v*h, C=C+k3c*h, k_vec=k_vec, epsilon=epsilon,
        k_const=k_const,potential=potential)
    k4r = v + k3v*h

    r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
    v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)
    C = C + (h/6) * (k1c + 2*k2c + 2*k3c + k4c)

    """
    # Boundary condition: particle cross the left boundary will appear on the other side
    r = np.where(r > box_dimension/2, r - box_dimension, r)
    r = np.where(r < -box_dimension/2, box_dimension - r, r)
    """

    H_em, H_mat = compute_H(
        q=q,r=r,v=v,k_vec=k_vec,C=C,epsilon=epsilon,k_const=k_const,potential=potential)

    mat_H_list.append(H_mat)
    em_H_list.append(H_em)

    H_list.append(H_mat + H_em)

    trajectory["r"].append(r)
    trajectory["v"].append(v)
    hamiltonian["em"].append(H_em)
    hamiltonian["mat"].append(H_mat)

    steps_list.append(i)
    if i % 1e2 == 0:
        print("Step {}".format(i+1))
        print("r = ",r)
        print("v = ",v)
        print("H_mat = ",H_mat)
        print("C = ",C)
        print("H_em = ",H_em)
        print("total H = ",H_mat + H_em)
        print("delta H_em / delta H_mat = ", 
            (em_H_list[-2] - H_em)/ (H_mat - mat_H_list[-2]) )


fig, ax = plt.subplots(3)

ax[0].plot(steps_list, em_H_list)
ax[0].set_ylabel(r"$H_{field}$")

ax[1].plot(steps_list, mat_H_list)
ax[1].set_ylabel(r"$H_{matter}$")

ax[2].plot(steps_list, H_list)
ax[2].set_ylim(np.array([-1e-2,1e-2]) + np.mean(H_list))
ax[2].set_xlabel("Time steps")
ax[2].set_ylabel(r"$H_{total}$")

fig.savefig("result_plot/particle_field_energy.jpeg",dpi=600)

with open("result_plot/trajectory.pkl","wb") as handle:
    pickle.dump(trajectory,handle)

with open("result_plot/hamiltonian.pkl","wb") as handle:
    pickle.dump(hamiltonian,handle)
