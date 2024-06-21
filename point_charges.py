import numpy as np
import matplotlib.pyplot as plt

from electromagnetic import FreeVectorPotential, CavityVectorPotential
import reduced_parameter as red

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

L = 1e4

##################################
### FREE FIELD POTENTIAL BEGIN ###
##################################
k_vector = np.array([
    [0,0,1],
    #[0,1,0],
    #[1,0,0],
    #[0,1,1],
    #[1,1,0],
    #[1,1,1]
    ]) * (2 * np.pi / L)

print(np.einsum("ki,ki->k",k_vector,k_vector) * red.c)

amplitude = np.vstack([
    #np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j,
    #np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j,
    np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j,
    #np.random.uniform(size = 2) * 10 + np.random.uniform(size = 2) * 10j,
    #np.random.uniform(size = 2) * 10 + np.random.uniform(size = 2) * 10j
    ]) * 100 * np.sqrt(L**3)

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
kappa = np.array([
        [0,1],
        [1,0],
        [1,1],
        ]) * (2 * np.pi / L)

m = np.array([1] * len(kappa))

amplitude = np.array([
    1 * np.random.uniform(size = 2) + 1j * np.random.uniform(size = 2),
    1 * np.random.uniform(size = 2) + 1j * np.random.uniform(size = 2),
    1 * np.random.uniform(size = 2) + 1j * np.random.uniform(size = 2),
    ]) * 100

Afield = CavityVectorPotential(
    kappa = kappa, m = m, amplitude = amplitude,
    L = L, S = L ** 2, constant_c = red.c)

### CAVITY POTENTIAL END ###
"""

#simple point charge
r = -np.array([[L, L, L]]) / 100
v = np.array([[1.0,1.0,1.0]])# * 1e3
q = np.array([np.eye(3)]) * 1e3

point_charge = PointCharges(q = q, r = r, r_dot = v)

t = 0
h = 1e-4

k = 100#(red.c * (2 * np.pi / L) ** 2) ** 2
print(np.sqrt(k))

Hmat = point_charge.kinetic_energy() + oscillator_potential(point_charge, k)
Hrad =  Afield.hamiltonian(True)

Hmat_list = [Hmat]
Hrad_list = [Hrad]
energy = [Hmat + Hrad]
time = [0]

#first iteration w/ Euler integration (and trapezoidal rule)
for i in range(5000):
    force_func = lambda t, charge_assembly:\
            EM_force(t, charge_assembly, Afield) \
            + oscillator_force(charge_assembly, k)
    
    point_charge.Verlet_step(t, h, force_func = force_func)
    
    #C_dot_t = Afield.dot_amplitude(t,point_charge)
    C_dot_tp1 = Afield.dot_amplitude(t+h,point_charge)
    C_new = Afield.C + h * (C_dot_tp1)

    t += h

    Afield.update_amplitude(C_new)

    Hmat = point_charge.kinetic_energy() + oscillator_potential(point_charge, k)
    Hrad = Afield.hamiltonian(True)

    energy.append(Hmat + Hrad)
    Hmat_list.append(Hmat)
    Hrad_list.append(Hrad)
    time.append(t)

fig,ax = plt.subplots()
ax.plot(time,energy)

fig.savefig("test.jpeg")

fig,ax = plt.subplots()
ax.plot(time,Hmat_list)

fig.savefig("test2.jpeg")

fig,ax = plt.subplots()
ax.plot(time,Hrad_list)

fig.savefig("test3.jpeg")






