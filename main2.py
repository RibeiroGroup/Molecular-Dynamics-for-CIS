import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
from Charge import ChargePoint, ChargeCluster

import constants

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
        m = 1, q = -1, 
        r = np.random.rand(3), # np.zeros(3), #
        v = np.random.rand(3), # np.zeros(3), # 
        )

beta = ChargePoint(
        m = 1, q = 1, 
        r = np.zeros(3), # np.random.rand(3), #
        v = np.random.rand(3), # np.zeros(3) #
        )

# EXPLICIT CALCULATION
def dot_C(q, r, v, k_vec, C, epsilon): 
    assert len(q) == len(v) and len(v) == len(r)

    k = np.sqrt(k_vec @ k_vec.T)
    omega = constants.c * k

    jk = 0
    for i,qi in enumerate(q):
        jk += np.exp(-1j * k_vec @ r[i]) * qi * v[i] # * (2 * np.pi)**(-1.5)

    jk_transv = (np.eye(3) - np.outer(k_vec, k_vec) / (k**2)) @ jk
    proj_jk_transv = np.array([
        jk_transv @ e for e in epsilon])

    return -1j * omega * C + \
        (2 * np.pi * 1j / k) * proj_jk_transv

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
            _ma_ += 0.5 * q[j] * q[l] * (r[j] - r[l] ) / d**3

        ma_list.append(_ma_)

    return np.array(ma_list)

def compute_morse_force():
    pass

def compute_oscillator_force(r, k_const):
    ma = []
    for ri in r:
        ma.append( - k_const * ri)
    return np.array(ma)

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
            K += 0.5 * q[i] * q[j] / d
    return K

def compute_Hmat(q,r,v):
    Hmat_transv = compute_Hmat_transv(q,r,v)
    Hmat_long = compute_Hmat_long(q,r,v)
    return Hmat_long + Hmat_transv

def compute_Hem(k_vec,C):
    return (2 * np.pi)**-1 * (k_vec @ k_vec.T) \
        * (C @ np.conjugate(C).T) 

def compute_H_oscillator(r,k_const):
    K = 0
    for ri in r:
        K += 0.5 * k_const * (ri @ ri.T).item()
    return K

def compute_force(q, r, v, k_vec, C, epsilon, k_const=None):
    F_transv = compute_transv_force(q, r, v, k_vec, C, epsilon)
    F_long = compute_long_force(q, r, v, k_vec, C, epsilon)
    F_oscillator = compute_oscillator_force(r,k_const) \
        if k_const!=None else 0
    return F_transv + F_long + F_oscillator

def compute_H(q, r, v, k_vec, C, epsilon, k_const=None):
    Hem = compute_Hem(k_vec, C)
    Hmat = compute_Hmat(q, r, v)
    if k_const != None:
        Hmat += compute_H_oscillator(r,k_const)
    return Hem, Hmat

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
q = [beta.q, alpha.q]
print("q = ",q)
r = np.vstack([beta.r, alpha.r])
r = r.reshape(-1,3)
print("r = ",r)
v = np.vstack([beta.v, alpha.v])
v = v.reshape(-1,3)
print("v = ",v)

print("############# others #############")

box_dimension = np.array([4]*3)
print("box dimension:", box_dimension)

k_const = None
print("oscillator constant k_const",k_const)

print("#############################################")

Hem, Hmat = compute_H(
        q=q,r=r,v=v,k_vec=k_vec,C=C,epsilon=epsilon,k_const=k_const)

print("Hamiltonian: {}(field) + {}(matter)".format(Hem,Hmat))

em_H_list = [Hem]
mat_H_list = [Hmat]
H_list = [Hem + Hmat]
steps_list = [0]

trajectory = {"initial":{"q":q,"r":r,"v":v,"k_const":k_const},
        "r":[r], "v":[v]}
hamiltonian = {"em":[Hem], "mat":[Hmat]}

for i in range(int(7e2+1)):
    k1c = dot_C(
        q=q, r=r, v=v, C=C, k_vec=k_vec, epsilon=epsilon)
    k1v = compute_force(
        q=q, r=r, v=v, C=C, k_vec=k_vec, epsilon=epsilon,k_const=k_const)
    k1r = v

    k2c = dot_C(
        q=q, r=r+k1r*h/2, v=v+k1v*h/2, C=C+k1c*h/2, k_vec=k_vec, epsilon=epsilon)
    k2v = compute_force(
        q=q, r=r+k1r*h/2, v=v+k1v*h/2, C=C+k1c*h/2, k_vec=k_vec, epsilon=epsilon,k_const=k_const)
    k2r = v + k1v*h/2

    k3c = dot_C(
        q=q, r=r+k2r*h/2, v=v+k2v*h/2, C=C+k2c*h/2, k_vec=k_vec, epsilon=epsilon)
    k3v = compute_force(
        q=q, r=r+k2r*h/2, v=v+k2v*h/2, C=C+k2c*h/2, k_vec=k_vec, epsilon=epsilon,k_const=k_const)
    k3r = v + k2v*h/2

    k4c = dot_C(
        q=q, r=r+k3r*h, v=v+k3v*h, C=C+k3c*h, k_vec=k_vec, epsilon=epsilon)
    k4v = compute_force(
        q=q, r=r+k3r*h, v=v+k3v*h, C=C+k3c*h, k_vec=k_vec, epsilon=epsilon,k_const=k_const)
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
        q=q,r=r,v=v,k_vec=k_vec,C=C,epsilon=epsilon,k_const=k_const)

    mat_H_list.append(H_mat)
    em_H_list.append(H_em)

    H_list.append(H_mat + H_em)

    trajectory["r"].append(r)
    trajectory["v"].append(v)
    hamiltonian["em"].append(H_em)
    hamiltonian["mat"].append(H_mat)

    steps_list.append(i)
    if i % 1e1 == 0:
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
#ax[2].set_ylim(np.array([-0.5,0.5]) + np.mean(H_list))
ax[2].set_xlabel("Time steps")
ax[2].set_ylabel(r"$H_{total}$")

fig.savefig("result_plot/particle_field_energy.jpeg",dpi=600)

with open("result_plot/trajectory.pkl","wb") as handle:
    pickle.dump(trajectory,handle)

with open("result_plot/hamiltonian.pkl","wb") as handle:
    pickle.dump(hamiltonian,handle)
