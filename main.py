from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from atoms import AtomsInBox
from electromagnetic import FreeVectorPotential,CavityVectorPotential
import reduced_parameter as red
from utils import EM_mode_generate

            ########################
            ########################
            ### EMPTY PARAMETERS ###
            ########################
            ########################
L = 1e5

t = 0
h = 1e-5

            ##########################
            ##########################
            ### INITIATE ATOMS BOX ###
            ##########################
            ##########################
atoms = AtomsInBox(box_length = L, cell_width = 1e3, mass_dict = red.mass_dict)
"""
np.random.seed(1)
atoms.random_initialize({"Ar":60, "Xe":60}, max_velocity = 100, min_velocity = 20)
"""
atoms.add(
        elements = ["Ar"],
        r = np.array([[-1,-1,-1]]),
        r_dot = np.array([[30,30,30]]),
        )

atoms.add(
        elements = ["Xe"],
        r = np.array([[1,1,1]]),
        #r_dot = np.array([[-10,-10,-10]]),
        r_dot = np.array([[-9,-9,-9]]),
        )

# Generate a matrix of LJ potential parameter
# e.g. matrix P with Pij is LJ parameter for i- and j-th atoms
idxAr = atoms.element_idx(element = "Xe")
idxXe = atoms.element_idx(element = "Ar")
epsilon_mat, sigma_mat = red.generate_LJparam_matrix(idxAr = idxAr, idxXe = idxXe)

# calculator to the atoms object
atoms.add_calculator(calculator_kwargs = {
    "epsilon": epsilon_mat, "sigma" : sigma_mat, 
    "positive_atom_idx" : idxXe, "negative_atom_idx" : idxAr,
    "mu0" : red.mu0 * 1e3, "d" : red.d0, "a" : red.a
    })

            ##########################
            ##########################
            ### INITIATE THE FIELD ###
            ##########################
            ##########################

k_vector = np.array(EM_mode_generate(max_n = 10, min_n = 0), dtype=np.float64)

np.random.seed(2024)

amplitude = np.vstack([
    np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j
    for i in range(len(k_vector))
    ]) * 1e-1 * np.sqrt(L**3)

##################################
### FREE FIELD POTENTIAL BEGIN ###
##################################
k_vector *= (2 * np.pi / L)
Afield = FreeVectorPotential(
        k_vector = k_vector, amplitude = amplitude,
        V = L ** 3, constant_c = red.c,
        )
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

### CAVITY POTENTIAL END ###
"""

            ############################
            ############################
            ### START THE SIMULATION ###
            ############################
            ############################

tlist = [t]
kinetic_elist = [atoms.kinetic()] 
potential_elist = [atoms.potential()] 
field_elist = [Afield.hamiltonian(return_sum_only=True)]

for i in tqdm(range(10000)):
    em_force_func = lambda t, atoms: Afield.force(t,atoms)

    atoms.Verlet_update(
            h = h, t = t,
            field_force = em_force_func
            )

    C_dot_tp1 = Afield.dot_amplitude(t+h,atoms)
    C_new = Afield.C + h * (C_dot_tp1)

    Afield.update_amplitude(C_new)
        
    t += h

    tlist.append(t)
    kinetic_elist.append(atoms.kinetic())
    potential_elist.append(atoms.potential())
    field_elist.append(Afield.hamiltonian(return_sum_only=True))


kinetic_elist = np.array(kinetic_elist)
potential_elist = np.array(potential_elist)
field_elist = np.array(field_elist)
total_energy = kinetic_elist + potential_elist + field_elist

print(total_energy[0])
print(total_energy[-1])

fig, ax = plt.subplots(2,2,figsize = (12,8))

ax[0,0].plot(tlist, total_energy)
ax[0,0].set_ylabel("Total energy")

ax[0,1].plot(tlist, kinetic_elist)
ax[0,1].set_ylabel("Kinetic energy")

ax[1,0].plot(tlist, potential_elist)
ax[1,0].set_ylabel("Potential energy")

ax[1,1].plot(tlist, field_elist)
ax[1,1].set_ylabel("Radiation energy")

fig.savefig("main_test.jpeg",dpi = 600)
