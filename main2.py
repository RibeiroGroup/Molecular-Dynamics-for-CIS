from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
from Charge import ChargePoint, ChargeCluster

import constants

np.random.seed(20202)
# FIELD SPECS
A = MultiModeField(
    C=1e-3* np.random.rand(2) + 1e-3j * np.random.rand(2),
    k=np.array([1/50,0,0]),
    epsilon=np.array([[0,1,0], [0,0,1]])
    )

#PARTICLE SPECS 
alpha = ChargePoint(
        m = 1, q = 1, 
        r = np.random.rand(3), 
        v = np.random.rand(3))

# EXPLICIT CALCULATION
print("### Initial field parameters value ###")
C = deepcopy(A.C[0])
print("C = ",C)
k_vec = deepcopy(A.k[0])
print("k = ",k_vec)
epsilon = deepcopy(A.epsilon[0])
print("epsilon = ",epsilon)
h = 1e-3
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

    k = k_vec @ k_vec.T
    omega = constants.c * (k_vec @ k_vec.T)

    return -1j * omega * C + \
        (2 * np.pi * 1j / k) * proj_jk_transv

def compute_force(q, r, v, k_vec, C, epsilon):
    C_dot = dot_C(q,r,v,k_vec,C,epsilon)

    _ma_ = np.array([0+0j,0+0j,0+0j])
    # k part
    _ma_[0] += v[1] * (1j * C[0] * np.exp(1j * k_vec @ r) \
              + np.conjugate(1j * C[0] * np.exp(1j * k_vec @ r)) )

    _ma_[0] += v[2] * (1j * C[1] * np.exp(1j * k_vec @ r) \
              + np.conjugate(1j * C[1] * np.exp(1j * k_vec @ r)) )

    # epsilon part
    for i in [1,2]:
        _ma_[i] += -C_dot[i-1] * np.exp(1j * k_vec @ r) + \
            np.conjugate(-C_dot[i-1] * np.exp(1j * k_vec @ r))

        _ma_[i] += -v[0] * ( 1j * C[i-1] * np.exp(1j * k_vec @ r) \
            + np.conjugate(1j * C[i-1] * np.exp(1j * k_vec @ r)) )

    _ma_ *= q / constants.c

    return _ma_

r = alpha.r
v = alpha.v

for i in range(100):
    print("Step {}".format(i+1))
    k1c = dot_C(
        q=alpha.q, r=r, v=v, 
        k_vec=k_vec, C=C, epsilon = epsilon)

    k1v = compute_force(
        q=alpha.q, r=r, v=v, 
        k_vec=k_vec, C=C, epsilon=epsilon)

    k1r = v

    k2c = dot_C(
        q=alpha.q, r=r + k1r/2, v=v + k1v/2, 
        k_vec=k_vec, C=C + k1c/2, epsilon = epsilon)

    k2v = compute_force(
        q=alpha.q, r=r + k1r/2, v=v + k1v/2, 
        k_vec=k_vec, C=C + k1c/2, epsilon=epsilon)

    k2r = v + k1v/2

    k3c = dot_C(
        q=alpha.q, r=r + k2r/2, v=v + k2v/2, 
        k_vec=k_vec, C=C + k2c/2, epsilon = epsilon)

    k3v = compute_force(
        q=alpha.q, r=r + k2r/2, v=v + k2v/2, 
        k_vec=k_vec, C=C + k2c/2, epsilon=epsilon)

    k3r = v + k2v/2

    k4c = dot_C(
        q=alpha.q, r=r + k3r, v=v + k3v, 
        k_vec=k_vec, C=C + k3c, epsilon = epsilon)

    k4v = compute_force(
        q=alpha.q, r=r + k3r, v=v + k3v, 
        k_vec=k_vec, C=C + k3c, epsilon=epsilon)

    k4r = v + k3v

    r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
    print("r = ",r)
    v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)
    print("v = ",v)
    C = C + (h/6) * (k1c + 2*k2c + 2*k3c + k4c)
    print("C = ",C)

