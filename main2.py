from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
from Charge import ChargePoint, ChargeCluster

import constants

np.random.seed(20202)
# FIELD SPECS
k_ = 2 * np.pi / (1000e-9 / constants.a0)
A = MultiModeField(
    C= 1 * np.random.rand(2) + 1j * np.random.rand(2),
    k=np.array([k_,0,0]),
    epsilon=np.array([[0,1,0], [0,0,1]])
    )

#PARTICLE SPECS 
alpha = ChargePoint(
        m = 1, q = 1, 
        r = np.zeros(3), # np.random.rand(3), 
        v = np.random.rand(3) * 1e-2, # np.zeros(3),
        )

# EXPLICIT CALCULATION
print("### Initial field parameters value ###")
C = deepcopy(A.C[0])
print("C = ",C)
k_vec = deepcopy(A.k[0])
print("k = ",k_vec)
epsilon = deepcopy(A.epsilon[0])
print("epsilon = ",epsilon)
h = 1e-4
print("h = ", h)
print("### Initial charge point parameters value ###")
print("q = ",alpha.q)
print("r = ",alpha.r)
print("v = ",alpha.v)
print("#######################################")

def dot_C(q, r, v, k_vec, C, epsilon): 
    jk = (2 * np.pi)**(-1.5) * np.exp(-1j * k_vec @ r) * q * v
    jk_transv = (np.eye(3) - np.outer(k_vec, k_vec)) @ jk
    proj_jk_transv = np.array([
        jk_transv @ e for e in epsilon])

    k = np.sqrt(k_vec @ k_vec.T)
    omega = constants.c * np.sqrt(k_vec @ k_vec.T)

    return -1j * omega * C + \
        (2 * np.pi * 1j / k) * proj_jk_transv

def compute_force(q, r, v, k_vec, C, epsilon):
    C_dot = dot_C(q,r,v,k_vec,C,epsilon)
    k = k_vec @ k_vec.T

    _ma_ = np.array([0+0j,0+0j,0+0j])
    # k part
    # C[0] = C_{k1}; C[1] = C_{k2}
    # v[0] = v_k, v[1] = v_{k1}, v[2] = v_{k2}
    _ma_[0] += v[1] * (1j * k * C[0] * np.exp(1j * k_vec @ r) \
        + np.conjugate(1j * k * C[0] * np.exp(1j * k_vec @ r)) )

    _ma_[0] += v[2] * (1j * k * C[1] * np.exp(1j * k_vec @ r) \
        + np.conjugate(1j * k * C[1] * np.exp(1j * k_vec @ r)) )

    # epsilon part
    for i in [1,2]:
        _ma_[i] +=       -C_dot[i-1] * np.exp(1j * k_vec @ r) + \
            np.conjugate(-C_dot[i-1] * np.exp(1j * k_vec @ r))

        _ma_[i] += -v[0] * (1j * k * C[i-1] * np.exp(1j * k_vec @ r) \
             + np.conjugate(1j * k * C[i-1] * np.exp(1j * k_vec @ r)) )

    _ma_ *= q / constants.c

    return _ma_

r = alpha.r
v = alpha.v

for i in range(int(1e6 + 1)):
    k1c = dot_C(
        q=alpha.q, r=r, v=v, 
        k_vec=k_vec, C=C, epsilon = epsilon)

    k1v = compute_force(
        q=alpha.q, r=r, v=v, 
        k_vec=k_vec, C=C, epsilon=epsilon)

    k1r = v

    k2c = dot_C(
        q=alpha.q, r=r + h*k1r/2, v=v + h*k1v/2, 
        k_vec=k_vec, C=C + h*k1c/2, epsilon = epsilon)

    k2v = compute_force(
        q=alpha.q, r=r + h*k1r/2, v=v + h*k1v/2, 
        k_vec=k_vec, C=C + h*k1c/2, epsilon=epsilon)

    k2r = v + h*k1v/2

    k3c = dot_C(
        q=alpha.q, r=r + h*k2r/2, v=v + h*k2v/2, 
        k_vec=k_vec, C=C + h*k2c/2, epsilon = epsilon)

    k3v = compute_force(
        q=alpha.q, r=r + h*k2r/2, v=v + h*k2v/2, 
        k_vec=k_vec, C=C + h*k2c/2, epsilon=epsilon)

    k3r = v + h*k2v/2

    k4c = dot_C(
        q=alpha.q, r=r + h*k3r, v=v + h*k3v, 
        k_vec=k_vec, C=C + h*k3c, epsilon = epsilon)

    k4v = compute_force(
        q=alpha.q, r=r + h*k3r, v=v + h*k3v, 
        k_vec=k_vec, C=C + h*k3c, epsilon=epsilon)

    k4r = v + h*k3v

    if i % 1e4 == 0:
        print("Step {}".format(i+1))

        r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
        print("r = ",r)

        v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)
        print("v = ",v)

        H_mat = 0.5 * v @ v.T

        C = C + (h/6) * (k1c + 2*k2c + 2*k3c + k4c)
        print("C = ",C)

        H_em = (2 * np.pi)**-1 * (k_vec @ k_vec.T) \
            * (C @ np.conjugate(C).T)

        print(H_mat + H_em)





