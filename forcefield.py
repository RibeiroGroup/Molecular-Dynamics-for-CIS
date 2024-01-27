import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import sympy as sm
from sympy.abc import a as sm_a
from sympy.abc import d as sm_D
from sympy.abc import r as sm_Re
from sympy.abc import x as sm_x, y as sm_y, z as sm_z

import constants
from simpleForceField_1 import MorsePotential as OldMorsePotential,\
    compute_Morse_force, compute_Hmorse

sm_Rx = sm.symbols("Rx")
sm_Ry = sm.symbols("Ry")
sm_Rz = sm.symbols("Rz")

class BasePotential:
    """
    Base class for molecular potential
    """
    def __init__(self):
        pass

    def get_Rij(self,Ri,Rj):
        assert Ri.shape[1] == 3 and Rj.shape[1] == 3
        return np.sqrt( np.sum( (Ri - Rj)**2, axis=1))

    def compute_force(r):
        force_list = []
        for i, ri in enumerate(r):
            ma = np.zeros(3)
            for j, rj in enumerate(r):
                if i == j: continue
                ma += self.force(center = rj, R = ri)

            force_list.append(ma)
        return np.array(force_list)

    def compute_potential(self,r):
        V = 0
        for i, ri in enumerate(r):
            for j, rj in enumerate(r):
                if i == j: continue
                # compute force on ri
                #=> rj is the center of Morse potential
                V += potential(center = rj, R = ri)
        return V/2

class MorsePotential(BasePotential):
    def __init__(self, De, Re, a):
        super().__init__()

        self.De = De
        self.Re = Re
        self.a = a

        self.potential = lambda Ri, Rj:\
                De*(1 - np.exp(-a*(self.get_Rij(Ri,Rj) - Re)))**2


    def force(self, Ri, Rj):

        force = 2*a*De*(1 - np.exp(-a*(-Re + self.get_Rij(Ri,Rj) ))) \
            *np.exp(-a*(-Re + self.get_Rij(Ri,Rj) )) \
            /self.get_Rij(Ri,Rj) 

        force = np.tile( force[:,np.newaxis],(1,3)) \
                * (-1.0*Rj + 1.0*Ri)

        return force

    def __call__(self,center,R):
        return self.morse(center, R)

    def get_analytical_form(self):

        Rij = ( (sm_x - sm_Rx)**2 \
                + (sm_y - sm_Ry)**2 \
                + (sm_z - sm_Rz)**2 )**0.5

        morse_exp = De*(1 - sm.exp(-a*(Rij - Re)))**2

        force_exp = [
            -sm.diff(self.morse_exp, sm_sym)
            for sm_sym in [sm_x, sm_y, sm_z]
                ]
        return morse_exp, force_exp


"""
De =  1495 / 4.35975e-18 / 6.023e23
Re = 3.5e-10 / 5.29177e-11
a = 1/ ( (1/3 * 1e-10) / 5.29177e-11)

potential1 = MorsePotential(De=De,Re=Re,a=a)
potential2 = OldMorsePotential(De=De,Re=Re,a=a)

np.random.seed(2024)
for j in range(5):
    Ri = np.random.rand(5,3)
    Rj = np.random.rand(5,3) * np.random.randint(1,10)

    print(potential1.force(Ri,Rj)- np.vstack(
        [potential2.compute_force(Ri[i],Rj[i])
            for i in range(5)
            ]
            ))

"""

"""

### TEST ###
De =  1495 / 4.35975e-18 / 6.023e23
Re = 3.5e-10 / 5.29177e-11
a = 1/ ( (1/3 * 1e-10) / 5.29177e-11)

print("a = ",a)
print("De = ",De)
print("Re = ",Re)

np.random.seed(2024)

h = 1e-2
n_particles = 2
r = np.array([[0,0,0],[10,0,0]],dtype=np.float64)
v = np.array([[1,0,0],[-1,0,0]],dtype=np.float64) * 5e-1

potential = MorsePotential(
        De=De, Re=Re, a=a)

#

center = np.zeros(3)
X = np.arange(6,20,0.1)
V = [potential(center, np.array([i,0,0])) for i in X]
F = np.array([potential.compute_force(center, np.array([i,0,0])) for i in X])
print(F)

fig, ax = plt.subplots()

ax.plot(X,V)
fig.savefig("morse_potential.jpeg",dpi=600)

fig, ax = plt.subplots()

ax.plot(X,F[:,0])
fig.savefig("morse_force.jpeg",dpi=600)

#

K = np.sum([0.5 * (vi @ vi.T) for vi in v])
V = compute_Hmorse(r,potential)
E = K + V
print("E = ", E)

trajectory = {"r":[r],"v":[v]}
K_list = [K]
V_list = [V]
E_list = [E]
dist = []

for step in range(1000+1):
    k1v = compute_Morse_force(r, potential)
    k1r = v

    k2v = compute_Morse_force(r + k1r * h/2, potential)
    k2r = v + k1v * h/2

    k3v = compute_Morse_force(r + k2r * h/2, potential)
    k3r = v + k2v * h/2

    k4v = compute_Morse_force(r + k3r * h, potential)
    k4r = v + k3v * h

    dr = (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
    r += dr

    dv = (h/6) * (k1v + 2*k2v + 2*k3v + k4v)
    v += dv

    K = np.sum([0.5 * (vi @ vi.T) for vi in v])
    V = compute_Hmorse(r,potential)
    E = K + V

    E_list.append(E)
    trajectory["r"].append(r)
    trajectory["v"].append(v)

    dist.append(np.sqrt((r[0]-r[1]) @ (r[0]-r[1]).T))
    if step % 100 == 0:
        print(step,", E = ", E)

fig, ax = plt.subplots()
ax.plot(
        np.arange(len(E_list)),
        E_list)
ax.set_ylim(np.mean(E_list) + np.array([-1e-1,1e-1]))

fig.savefig("Morse_energy.jpeg",dpi=600)

fig, ax = plt.subplots()
ax.plot(
        np.arange(len(dist)),
        dist)

fig.savefig("Morse_dist.jpeg",dpi=600)

"""
