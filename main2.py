from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
from Charge import ChargePoint, ChargeCluster

import constants

np.random.seed(2024)
# FIELD SPECS
k_ = 2 * np.pi / (1e-9 / constants.a0)
A = MultiModeField(
    C= (np.random.rand(2) + 1j * np.random.rand(2)),
    k=np.array([k_,0,0]),
    epsilon=np.array([[0,1,0], [0,0,1]])
    )

#PARTICLE SPECS 
alpha = ChargePoint(
        m = 1, q = 1, 
        r = np.random.rand(3), # np.zeros(3), # 
        v = np.random.rand(3), # np.zeros(3), # 
        )

# EXPLICIT CALCULATION
print("### Initial field parameters value ###")
C = deepcopy(A.C[0])
print("C = ",C)
k_vec = deepcopy(A.k[0])
print("k = ",k_vec)
epsilon = np.array(A.epsilon[0])
print("epsilon = ",epsilon)
h = 1e-4
print("h = ", h)
print("### Initial charge point parameters value ###")
print("q = ",alpha.q)
print("r = ",alpha.r)
print("v = ",alpha.v)
print("#######################################")

def dot_C(q, r, v, k_vec, C, epsilon): 
    k = np.sqrt(k_vec @ k_vec.T)
    omega = constants.c * k

    jk = np.exp(-1j * k_vec @ r) * q * v #* (1 * np.pi)**(-1.5)
    jk_transv = (np.eye(3) - np.outer(k_vec, k_vec) / (k**2)) @ jk
    proj_jk_transv = np.array([
        jk_transv @ e for e in epsilon])

    return -1j * omega * C + \
        (2 * np.pi * 1j / k) * proj_jk_transv

def compute_force(q, r, v, k_vec, C, epsilon):
    C_dot = dot_C(q,r,v,k_vec,C,epsilon)

    k = np.sqrt(k_vec @ k_vec.T)

    _ma_ = np.array([0+0j,0+0j,0+0j])
    # k part
    # C[0] = C_{k1}; C[1] = C_{k2}
    vk1 = epsilon[0] @ v.T ; vk2 = epsilon[1] @ v.T

    _ma_[0] += vk1 * (1j * k * C[0] * np.exp(1j * k_vec @ r) \
        + np.conjugate(1j * k * C[0] * np.exp(1j * k_vec @ r)) )

    _ma_[0] += vk2 * (1j * k * C[1] * np.exp(1j * k_vec @ r) \
        + np.conjugate(1j * k * C[1] * np.exp(1j * k_vec @ r)) )

    # epsilon part
    for i in [1,2]:
        _ma_[i] +=       -C_dot[i-1] * np.exp(1j * k_vec @ r) + \
            np.conjugate(-C_dot[i-1] * np.exp(1j * k_vec @ r))

        _ma_[i] += -v[0] * (1j * k * C[i-1] * np.exp(1j * k_vec @ r) \
             + np.conjugate(1j * k * C[i-1] * np.exp(1j * k_vec @ r)) )

    _ma_ *= q / constants.c

    return _ma_

def compute_Hmat(v):
    return 0.5 * v @ v.T

def compute_Hem(k_vec,C):
    return (2 * np.pi)**-1 * (k_vec @ k_vec.T) \
        * (C @ np.conjugate(C).T) 

r = alpha.r
v = alpha.v

steps_list = [0]
em_H_list = [compute_Hem(k_vec, C)]
mat_H_list = [compute_Hmat(v)]
H_list = [compute_Hem(k_vec, C) + compute_Hmat(v)]

for i in range(int(1e4 + 1)):
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

    r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
    v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)
    C = C + (h/6) * (k1c + 2*k2c + 2*k3c + k4c)

    H_mat = compute_Hmat(v)# * constants.c
    mat_H_list.append(H_mat)

    H_em = compute_Hem(k_vec, C) 
    em_H_list.append(H_em)

    H_list.append(H_mat + H_em)

    steps_list.append(i)
    if i % 1e3 == 0:
        print("Step {}".format(i+1))
        print("r = ",r)
        print("v = ",v)
        print("H_mat = ",H_mat)
        print("C = ",C)
        print("H_em = ",H_em)
        print("total H = ",H_mat + H_em)
        print("delta H_em / delta H_mat = ", 
            (H_em - em_H_list[-2]) / (H_mat - mat_H_list[-2]) )

fig, ax = plt.subplots(3)

ax[0].plot(steps_list, em_H_list)

ax[1].plot(steps_list, mat_H_list)

ax[2].plot(steps_list, H_list)
ax[2].set_ylim(np.array([-0.5,0.5]) + np.mean(H_list))

fig.savefig("result_plot\\particle_field_energy.jpeg",dpi=600)



