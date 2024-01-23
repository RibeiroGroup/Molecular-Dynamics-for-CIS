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
k_ = 1/constants.c# 2 * np.pi / (100e-9 / constants.a0)
A = MultiModeField(
    C= (np.random.rand(2) + 1j * np.random.rand(2)) * 1e1,
    k=np.array([k_,0,0]),
    epsilon=np.array([[0,1,0], [0,0,1]])
    )

#PARTICLE SPECS 
alpha = ChargePoint(
        m = 1, q = 1, 
        r = np.ones(3) * 7,
        v = np.ones(3) * 1e-2 # np.random.rand(3), # np.zeros(3), # 
        )

beta = ChargePoint(
        m = 1, q = 1, 
        r = np.ones(3) * 2, #np.zeros(3), # np.random.rand(3), #
        v = np.zeros(3), #np.ones(3) * 2
        )

####################################################################
####################################################################
####################################################################
print("####### Initial field parameters value #######")
####################################################################
C = A.C[0]
#C = np.zeros(2)
print("C = ",C)
k_vec = deepcopy(A.k[0])
print("k = ",k_vec)
epsilon = np.array(A.epsilon[0])
print("epsilon = ",epsilon)

####################################################################
print("### Initial charge point parameters value ###")
####################################################################
q = [beta.q]#, alpha.q]
print("q = ",q)
r = np.vstack([beta.r])#, alpha.r])
r = r.reshape(-1,3)
print("r = ",r)
v = np.vstack([beta.v])#, alpha.v])
v = v.reshape(-1,3)
print("v = ",v)

####################################################################
print("############# Potential and oscillators parameters #############")
####################################################################

k_const = 1/constants.c
print("oscillator constant k_const",k_const)

De = 1495 / 4.35975e-18 / 6.023e23
Re = (3.5e-10) / 5.29177e-11
a = 1/ ( (1/3 * 1e-10) / 5.29177e-11)

potential = None # MorsePotential(De=De, Re=Re, a=a)

####################################################################
print("############# simulation environmental parameters #############")
####################################################################
h = 1e-2
print("h = ", h)

box_dimension = np.array([4]*3)
print("box dimension:", box_dimension)

print("#############################################")
####################################################################
####################################################################
####################################################################

md_sim = SimpleDynamicModel(
    q = q, k_vec = k_vec, epsilon = epsilon, 
    k_const = k_const, potential = potential,
)

md_sim_2 = SimpleDynamicModel(
    q = q, k_vec = k_vec, epsilon = epsilon, 
    k_const = k_const, potential = potential,
    exclude_EM = True
)
r2 = deepcopy(r)
v2 = deepcopy(v)
C2 = np.zeros(2)

Hem, Hmat, H_osci = md_sim.compute_H(r=r, v=v, C=C)

print("Hamiltonian: {}(field) + {}(matter)".format(Hem,Hmat))

H_list = [Hem + Hmat + H_osci]
steps_list = [0]

trajectory = {"initial":{"q":q,"r":r,"v":v,"k_const":k_const},
        "r":[r], "v":[v]}
hamiltonian = {"em":[Hem], "mat":[Hmat], "osci":[H_osci]}
amplitude = [np.sqrt(r @ r.T)[0][0]]

Hem, Hmat, H_osci = md_sim_2.compute_H(r=r, v=v, C=C)
H_list2 = [Hem + Hmat + H_osci]
hamiltonian2 = {"em":[Hem], "mat":[Hmat], "osci":[H_osci]}
amplitude2 = [np.sqrt(r2 @ r2.T)[0][0]]

for i in range(int(15e3+1)):
    r,v,C = md_sim.rk4_step(r=r,v=v,C=C,h=h)
    Hem, Hmat, H_osci = md_sim.compute_H(r=r, v=v, C=C)

    trajectory["r"].append(r)
    trajectory["v"].append(v)

    H_list.append(Hmat + Hem + H_osci)
    hamiltonian["em"].append(Hem)
    hamiltonian["mat"].append(Hmat)
    hamiltonian["osci"].append(H_osci)
    amplitude.append(np.sqrt(r @ r.T)[0][0])

    steps_list.append(i)
    if i % 1e2 == 0:
        print("Step {}".format(i+1))
        print("r = ",r)
        print("v = ",v)
        print("H_mat = ",Hmat)
        print("C = ",C)
        print("H_em = ",Hem)
        print("total H = ",Hmat + Hem + H_osci)

    r2,v2,C2 = md_sim_2.rk4_step(r=r2,v=v2,C=C2,h=h)
    Hem, Hmat, H_osci = md_sim_2.compute_H(r=r2, v=v2, C=C2)

    H_list2.append(Hmat + Hem + H_osci)
    hamiltonian2["em"].append(Hem)
    hamiltonian2["mat"].append(Hmat)
    hamiltonian2["osci"].append(H_osci)
    amplitude2.append(np.sqrt(r2 @ r2.T)[0][0])

steps_list = np.array(steps_list) * h
fig, ax = plt.subplots(2,3,figsize= (18,6))

ax[0,0].plot(steps_list, hamiltonian["em"],c="c")
ax[0,0].set_ylabel(r"$H_{field}$")

ax[1,0].plot(steps_list, hamiltonian["mat"],c="c")
ax[1,0].plot(steps_list, hamiltonian2["mat"],c="r")
ax[1,0].set_ylabel(r"$H_{matter}$")
ax[1,0].set_xlabel("Absolute time")

ax[0,1].plot(steps_list, H_list,c="c")
ax[0,1].plot(steps_list, H_list2,c="r")
#ax[0,1].set_ylim(np.array([-1e-3,1e-3]) + np.mean(H_list))
ax[0,1].set_ylabel(r"$H_{total}$")

ax[1,1].plot(steps_list, hamiltonian["osci"],c="c")
ax[1,1].plot(steps_list, hamiltonian2["osci"],c="r")
ax[1,1].set_ylabel(r"$H_{oscillator}$")
ax[1,1].set_xlabel("Absolute time")

ax[1,2].plot(steps_list, amplitude,c="c")
ax[1,2].plot(steps_list, amplitude2,c="r")
ax[1,2].set_ylabel(r"Amplitude $|x|$")
ax[1,2].set_xlabel("Absolute time")

ax[0,2].plot([],[],c="c",label = "HO in EM field")
ax[0,2].plot([],[],c="r",label = "Ordinary HO")
ax[0,2].legend()

fig.savefig("result_plot/particle_field_energy.jpeg",dpi=600,bbox_inches="tight")

with open("result_plot/trajectory.pkl","wb") as handle:
    pickle.dump(trajectory,handle)

with open("result_plot/hamiltonian.pkl","wb") as handle:
    pickle.dump(hamiltonian,handle)
