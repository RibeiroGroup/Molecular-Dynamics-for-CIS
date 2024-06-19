import numpy as np
import matplotlib.pyplot as plt

from electromagnetic import FreeFieldVectorPotential
import reduced_parameter as red

"""
Testing electromagnetic.py
"""

k_vector = np.array([
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,1,0],
    [1,1,1]
    ]) / (red.c)
amplitude = np.vstack([
    np.random.uniform(size = 2) * 10 + np.random.uniform(size = 2) * 10j,
    np.random.uniform(size = 2) * 10 + np.random.uniform(size = 2) * 10j,
    np.random.uniform(size = 2) * 10 + np.random.uniform(size = 2) * 10j,
    np.random.uniform(size = 2) * 10 + np.random.uniform(size = 2) * 10j,
    np.random.uniform(size = 2) * 10 + np.random.uniform(size = 2) * 10j
    ])

Afield = FreeFieldVectorPotential(
        k_vector = k_vector, amplitude = amplitude,
        V = 1.0, constant_c = red.c,
        )

"""
print(Afield.k_vector)
print(Afield.pol_vec)
print(Afield.k_val)
print(Afield.omega)
raise Exception
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

    def current(self, k_vec, mode_function = None):
        qr_dot = np.einsum("nij,ni->nj", q, self.r_dot)

        if mode_function == None:
            exp_ikr = np.exp(
                np.einsum("ki,ni->kn", -1j * k_vec, self.r)) # i = 3

        elif mode_function == "TE":
            raise Exception("To be implemented")

        elif mode_function == "TM":
            raise Exception("To be implemented")

        Jk = np.einsum("nj,kn->kj",qr_dot,exp_ikr) # j = 3

        return Jk

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

#simple point charge
r = np.array([[1.0,1.0,1.0]])
v = np.array([[1.0,0.0,0.0]]) * 10
q = np.array([np.eye(3)])

point_charge = PointCharges(q = q, r = r, r_dot = v)

t = 0
h = 1e-3

Hmat = point_charge.kinetic_energy()
Hrad =  Afield.hamiltonian(True)

Hmat_list = [Hmat]
Hrad_list = [Hrad]
energy = [Hmat + Hrad]
time = [0]

#first iteration w/ Euler integration (and trapezoidal rule)

for i in range(3000):
    force_func = lambda t, charge_assembly: EM_force(t, charge_assembly, Afield)
    
    point_charge.Verlet_step(t, h, force_func = force_func)
    
    C_dot = Afield.dot_amplitude(t+h,point_charge)
    C_new = Afield.C + h * C_dot

    t += h

    Afield.update_amplitude(C_new)

    Hmat = point_charge.kinetic_energy()
    Hrad = Afield.hamiltonian(True)

    energy.append(Hmat + Hrad)
    Hmat_list.append(Hmat)
    Hrad_list.append(Hrad)
    time.append(t)

print(Afield.k_vector)
print(Afield.pol_vec)
print(Afield.k_val)
print(Afield.omega)

fig,ax = plt.subplots()
ax.plot(time,energy)

fig.savefig("test.jpeg")

fig,ax = plt.subplots()
ax.plot(time,Hmat_list)

fig.savefig("test2.jpeg")

fig,ax = plt.subplots()
ax.plot(time,Hrad_list)

fig.savefig("test3.jpeg")






