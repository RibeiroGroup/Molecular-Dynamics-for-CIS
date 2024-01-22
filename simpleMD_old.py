import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
from Charge import ChargePoint, ChargeCluster
from simpleForceField import MorsePotential, compute_Morse_force, compute_Hmorse

import constants
# EXPLICIT CALCULATION
def dot_C(q, r, v, k_vec, C, epsilon): 
    assert len(q) == len(v) and len(v) == len(r)

    k = np.sqrt(k_vec @ k_vec.T)
    omega = constants.c * k

    jk = 0
    for i,qi in enumerate(q):
        jk += np.exp(-1j * k_vec @ r[i]) * qi * v[i]# * (2 * np.pi)**(-1.5)

    jk_transv = (np.eye(3) - np.outer(k_vec, k_vec) / (k**2)) @ jk
    proj_jk_transv = np.array([
        jk_transv @ e for e in epsilon])

    return -1j * omega * C + \
        (2 * np.pi * 1j / k) * proj_jk_transv

#####################################################
### COMPUTE FORCE ###################################
#####################################################

def compute_transv_force(q, r, v, k_vec, C, epsilon):
    assert len(q) == len(v) and len(v) == len(r)
    C_dot = dot_C(q,r,v,k_vec,C,epsilon)

    k = np.sqrt(k_vec @ k_vec.T)
    epsilon_k = k_vec / k

    ma_list = []

    for j, rj in enumerate(r):
        _ma_ = np.array([0+0j,0+0j,0+0j])
        # k part
        vk = epsilon_k @ v[j].T   # projection of v on k_vec
        vk1 = epsilon[0] @ v[j].T # projection of v on epsilon_k1
        vk2 = epsilon[1] @ v[j].T # projection of v on epsilon_k2

        # C[0] = 0;  C[1] = C_{k1}; C[2] = C_{k2}
        _ma_k = vk1 * (1j * k * C[0] * np.exp(1j * k_vec @ rj) \
           + np.conjugate(1j * k * C[0] * np.exp(1j * k_vec @ rj)) )

        _ma_k += vk2 * (1j * k * C[1] * np.exp(1j * k_vec @ rj) \
           + np.conjugate(1j * k * C[1] * np.exp(1j * k_vec @ rj)) )

        _ma_ += _ma_k * epsilon_k

        # epsilon part
        for i in [1,2]:
            _ma_ki =         -C_dot[i-1] * np.exp(1j * k_vec @ rj) + \
                np.conjugate(-C_dot[i-1] * np.exp(1j * k_vec @ rj))

            _ma_ki += -vk * (1j * k * C[i-1] * np.exp(1j * k_vec @ rj) \
              + np.conjugate(1j * k * C[i-1] * np.exp(1j * k_vec @ rj)) )

            _ma_ += _ma_ki * epsilon[i-1]

        _ma_ *= q[j] / constants.c

        ma_list.append(np.real(_ma_))

    return np.array(ma_list)

def compute_long_force(q, r, v, k_vec, C, epsilon):
    ma_list = []
    for j, rj in enumerate(r):
        _ma_ = np.zeros(3) + 1.j * np.zeros(3)
        for l, rl in enumerate(r):
            if l == j: continue
            d = np.sqrt(np.sum( (r[l] - r[j])**2 ))
            _ma_ +=  q[j] * q[l] * (r[j] - r[l] ) / d**3

        ma_list.append(_ma_)

    return np.array(ma_list)

def compute_oscillator_force(r, k_const):
    ma = []
    for ri in r:
        ma.append( - k_const * ri)
    return np.array(ma)

#####################################################
### COMPUTE HAMILTONIAN #############################
#####################################################

def compute_Hmat_transv(q,r,v):
    K = 0
    for i, vi in enumerate(v):
        K += 0.5 * vi @ vi.T
    return K

def compute_Hmat_long(q,r,v):
    K = 0
    for i, vi in enumerate(v):
        if i == len(v) - 1: continue
        for j in range(len(v)):
            if i == j: continue
            d = np.sqrt( np.sum( (r[i] - r[j])**2 ) )
            K += q[i] * q[j] / d
    return K

def compute_Hem(k_vec,C):
    return (2 * np.pi)**-1 * (k_vec @ k_vec.T) \
        * (C @ np.conjugate(C).T) 

def compute_H_oscillator(r,k_const):
    K = 0
    for ri in r:
        K += 0.5 * k_const * (ri @ ri.T).item()
    return K

#####################################################
### WRAPPER #########################################
#####################################################

def compute_force(q, r, v, k_vec, C, epsilon, k_const=None, potential=None):
    F_transv = compute_transv_force(q, r, v, k_vec, C, epsilon)
    F_long = compute_long_force(q, r, v, k_vec, C, epsilon)
    if potential:
        F_long = compute_Morse_force(r,potential)
    F_oscillator = compute_oscillator_force(r,k_const) \
        if k_const!=None else 0
    return F_transv + F_long + F_oscillator

def compute_Hmat(q,r,v,potential=None):
    Hmat_transv = compute_Hmat_transv(q,r,v)
    Hmat_long = compute_Hmat_long(q,r,v)
    if potential:
        Hmat_long = compute_Hmorse(r,potential)
    return Hmat_long + Hmat_transv

def compute_H(q, r, v, k_vec, C, epsilon, k_const=None,potential=None):
    Hem = compute_Hem(k_vec, C)
    Hmat = compute_Hmat(q, r, v, potential)
    if k_const != None:
        Hmat += compute_H_oscillator(r,k_const)
    return Hem, Hmat

