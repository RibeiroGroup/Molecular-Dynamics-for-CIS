from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from electromagnetic import FreeVectorPotential, CavityVectorPotential
import reduced_parameter as red
from utils import EM_mode_generate

"""
Testing electromagnetic.py
Demo the electromagnetic.py for simulating the simplest 
system: one point charge 
"""

class PointCharges:
    def __init__(self, q, r, r_dot):
        self.N = len(q)

        self.q = q
        self.update(r, r_dot)

    def update(self, r, r_dot):
        assert len(r) == self.N
        assert len(r_dot) == self.N
        self.r = r
        self.r_dot = r_dot

    def current_mode_projection(self):

        qr_dot = np.einsum("nij,ni->nj", q, self.r_dot)

        return qr_dot

    def Verlet_step(self, t, h, force_func):
        force = force_func(t, self)

        v_half = self.r_dot + force * h / 2
        r_new = self.r + v_half * h

        self.update(r_new, v_half)

        new_force = force_func(t + h, self)
        v_new = v_half + new_force * h / 2

        self.update(r = r_new, r_dot = v_new)

    def kinetic_energy(self):
        k = 0.5 * np.einsum("ni,ni->n",self.r_dot,self.r_dot)
        return np.sum(k)

def oscillator_force(charge_assemble, k):
    r = charge_assemble.r
    return - k * r

def oscillator_potential(charge_assemble, k):
    r = charge_assemble.r
    return 0.5 * k * np.sum(np.einsum("ni,ni->n",r,r))
         
def EM_force(t, charge_assemble , A):

    dAdt = A.time_diff(t,charge_assemble)
    gradA = A.gradient(t,charge_assemble.r)
    r_dot = charge_assemble.r_dot
    q = charge_assemble.q

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

def profiling_rad(omega_list,unique_omega,Hrad):
    rad_profile = []

    for i, omega in enumerate(unique_omega):
        rad_profile.append(
                np.sum(Hrad[omega_list == omega])
                )

    return rad_profile

L = 1e8

k_vector = np.array(EM_mode_generate(max_n = 10, min_n = 1), dtype=np.float64)

print(k_vector.shape)

np.random.seed(2024)
amplitude = np.vstack([
    np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j
    for i in range(len(k_vector))
    ]) * 0e0 * np.sqrt(L**3)


##################################
### FREE FIELD POTENTIAL BEGIN ###
##################################
k_vector *= (2 * np.pi / L)
Afield = FreeVectorPotential(
        k_vector = k_vector, amplitude = amplitude,
        V = L ** 3, constant_c = red.c,
        )
print("Warning, the volume is set to 1")
### FREE FIELD POTENTIAL END ###
"""
##############################
### CAVITY POTENTIAL BEGIN ###
##############################
kappa = k_vector[:,:2] * (2 * np.pi / L)

m = k_vector[:,-1].reshape(-1)

Afield = CavityVectorPotential(
    kappa = kappa, m = m, amplitude = amplitude,
    L = L, S = L ** 2, constant_c = red.c)

print(Afield.kappa_unit.shape)

### CAVITY POTENTIAL END ###
"""

omega_list = red.c * np.sqrt(np.einsum("ki,ki->k",k_vector, k_vector))
unique_omega = list(set(omega_list))

#simple point charge
r = -np.array([[1, 1, 1]]) * 100.0
v = np.array([[1.0,1.0,1.0]]) #* 1e2
q = np.array([np.eye(3)]) * 1e4

point_charge = PointCharges(q = q, r = r, r_dot = v)

t = 0
h = 1e-4

k = 100#(red.c * (2 * np.pi / L) ** 2) ** 2
print(np.sqrt(k))

Hmat = point_charge.kinetic_energy() + oscillator_potential(point_charge, k)
Hrad =  Afield.hamiltonian(False)
total_Hrad = np.sum(Hrad)

Hmat_list = [Hmat]
Hrad_list = [total_Hrad]

energy = [Hmat + total_Hrad]
rad_profile = profiling_rad(omega_list,unique_omega,Hrad)
rad_profile = [rad_profile]
time = [0]

#first iteration w/ Euler integration (and trapezoidal rule)
for i in tqdm(range(20000)):
    
    force_func = lambda t, charge_assembly:\
            EM_force(t, charge_assembly, Afield) \
            + oscillator_force(charge_assembly, k)

    point_charge.Verlet_step(t, h, force_func = force_func)
    
    #C_dot_t = Afield.dot_amplitude(t,point_charge)
    C_dot_tp1 = Afield.dot_amplitude(t+h,point_charge)
    C_new = Afield.C + h * (C_dot_tp1 )

    Afield.update_amplitude(C_new)
        
    t += h

    Hmat = point_charge.kinetic_energy() + oscillator_potential(point_charge, k)
    Hrad = Afield.hamiltonian(False)
    total_Hrad = np.sum(Hrad)

    energy.append(Hmat + total_Hrad)
    Hmat_list.append(Hmat)
    Hrad_list.append(total_Hrad)

    rad_profile.append(profiling_rad(omega_list,unique_omega,Hrad))

    time.append(t)

fig,ax = plt.subplots(2,2,figsize = (12,8))
ax[0,0].plot(time,energy)
ax[0,0].set_ylabel("Total energy")
ax[0,1].plot(time,Hmat_list)
ax[0,1].set_ylabel("Matter Hamiltonian")
ax[1,0].plot(time,Hrad_list)
ax[1,0].set_ylabel("Radiation Hamiltonian")

unique_omega = list(unique_omega)
rad_profile = np.max(np.array(rad_profile),axis=0).reshape(-1)

ax[1,1].scatter(unique_omega,rad_profile)
ax[1,1].set_ylabel("Radiation Hamiltonian")

fig.savefig("test.jpeg",dpi = 600,bbox_inches = "tight")






