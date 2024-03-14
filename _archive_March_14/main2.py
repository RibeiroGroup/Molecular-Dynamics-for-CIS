import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
from simpleForceField import MorsePotential, compute_Morse_force, compute_Hmorse

import constants
from simpleMD import *

np.random.seed(20)
# FIELD SPECS
n_modes = 4
k_ = 1/constants.c# 2 * np.pi / (100e-9 / constants.a0)
A = MultiModeField(
    C = np.tile((np.random.rand(1,2) + 1j * np.random.rand(1,2)),(n_modes,1)),
    k = np.array([
        [k_,0,0],
        #[k_,0,0],
        [0,k_,0],
        [0,0,k_],
        [k_,k_,0]]),
    epsilon = np.array([
        [[0,1,0], [0,0,1]],
        #[[0,1,0], [0,0,1]],
        [[1,0,0], [0,0,1]],
        [[0,1,0], [1,0,0]],
        [[0,0,1], [2**(-0.5),-2**(-0.5),0.0]],
        ], dtype=np.float64)
    )

####################################################################
####################################################################
####################################################################
print("####### Initial field parameters value #######")
####################################################################
C = A.C[:n_modes].reshape(-1,2)
#C = np.zeros(2)
print("C = ",C)

k_vec = deepcopy(A.k)[:n_modes]
print("k = ",k_vec)

epsilon = A.epsilon[:n_modes]
print("epsilon = ",epsilon)

####################################################################
print("### Initial charge point parameters value ###")
####################################################################
q = [1]#,1]
print("q = ",q)
r = np.vstack([
        #np.ones(3) * 2,
        np.zeros(3),
    ])
r = r.reshape(-1,3)
print("r = ",r)
v = np.vstack([
        [10,0,0]
        #np.zeros(3),
        #np.ones(3)
    ])
v = v.reshape(-1,3)
print("v = ",v)

####################################################################
print("############# Potential and oscillators parameters #############")
####################################################################

k_const = None
print("oscillator constant k_const",k_const)

De = 1495 / 4.35975e-18 / 6.023e23
Re = (3.5e-10) / 5.29177e-11
a = 1/ ( (1/3 * 1e-10) / 5.29177e-11)

potential = None # MorsePotential(De=De, Re=Re, a=a)

####################################################################
print("############# simulation environmental parameters #############")
####################################################################
h = 1e-3
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

Hem, Hmat, H_osci = md_sim.compute_H(r=r, v=v, C=C)

sim_result = {
        "initial":{"k_vec":k_vec,"k_const":k_const},
        "C":[C],
        "r":[r], "v":[v], "steps":[0], "h" : h,
        "em":[Hem], "mat":[Hmat], "osci":[H_osci],
        "amplitude": [np.sqrt(r @ r.T)[0][0]]
        }

"""
md_sim_2 = SimpleDynamicModel(
    q = q, k_vec = k_vec, epsilon = epsilon, 
    k_const = k_const, potential = potential,
    exclude_EM = True
)
r2 = deepcopy(r)
v2 = deepcopy(v)
C2 = deepcopy(C)

Hem, Hmat, H_osci = md_sim_2.compute_H(r=r, v=v, C=C)
H_list2 = [Hem + Hmat + H_osci]
sim_result2 = {
        "initial":{"q":q,"r":r2,"v":v2,"k_const":k_const},
        "r":[r], "v":[v], "steps":[0], "h" : h,
        "em":[Hem], "mat":[Hmat], "osci":[H_osci],
        "amplitude": [np.sqrt(r2 @ r2.T)[0][0]]
        }

"""
for i in range(int(10e3+1)):
    r,v,C = md_sim.rk4_step(r=r,v=v,C=C,h=h)
    Hem, Hmat, H_osci = md_sim.compute_H(r=r, v=v, C=C)

    sim_result["r"].append(r)
    sim_result["v"].append(v)

    sim_result["em"].append(Hem)
    sim_result["mat"].append(Hmat)
    sim_result["osci"].append(H_osci)
    sim_result["steps"].append(i+1)
    sim_result["amplitude"].append(np.sqrt(r @ r.T)[0][0])
    sim_result["C"].append(C)

    if i % 1e2 == 0:
        print("Step {}".format(i+1))
        print("r = ",r)
        print("v = ",v)
        print("H_mat = ",Hmat)
        print("C = ",C)
        print("H_em = ",Hem)
        print("total H = ",Hmat + np.sum(Hem) + H_osci)

    """
    r2,v2,C2 = md_sim_2.rk4_step(r=r2,v=v2,C=C2,h=h)
    Hem, Hmat, H_osci = md_sim_2.compute_H(r=r2, v=v2, C=C2)

    sim_result2["r"].append(r2)
    sim_result2["v"].append(v2)

    sim_result2["em"].append(Hem)
    sim_result2["mat"].append(Hmat)
    sim_result2["osci"].append(H_osci)
    sim_result2["amplitude"].append(np.sqrt(r2 @ r2.T)[0][0])

    """
with open("result_plot/sim_result.pkl","wb") as handle:
    pickle.dump(sim_result,handle)

"""
with open("result_plot/sim_result2.pkl","wb") as handle:
    pickle.dump(sim_result2,handle)
"""
