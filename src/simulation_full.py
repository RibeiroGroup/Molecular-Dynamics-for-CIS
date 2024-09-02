from tqdm import tqdm
import pickle

import numpy as np
import matplotlib.pyplot as plt

from calculator.calculator import Calculator

from matter.atoms import AtomsInBox

from field.electromagnetic import FreeVectorPotential,CavityVectorPotential

import utilities.reduced_parameter as red

"""
Test the model's energy conseravation by simulating a single collision
"""

            ########################
            ########################
            ### EMPTY PARAMETERS ###
            ########################
            ########################
L = 3e7
Lz = 3e3

t = 0
h = 1e-3

            ##########################
            ##########################
            ### INITIATE ATOMS BOX ###
            ##########################
            ##########################
atoms = AtomsInBox(
    Lxy = L, Lz = Lz, cell_width = (1e6,1e2), 
    mass_dict = red.mass_dict)

atoms.add(
        elements = ["Ar"],
        r = np.array([[-0.25,-1.00,-1.0]]),
        r_dot = np.array([[0.0,0.3,0.3]]) / np.sqrt(2),
        )

atoms.add(
        elements = ["Xe"],
        r = np.array([[0.00,2.00,2.0]]),
        r_dot = np.array([[-0.00,-0.09,-0.09]]) / np.sqrt(2),
        )

# Generate a matrix of LJ potential parameter
# e.g. matrix P with Pij is LJ parameter for i- and j-th atoms
idxAr = atoms.element_idx(element = "Xe")
idxXe = atoms.element_idx(element = "Ar")
epsilon_mat, sigma_mat = red.generate_LJparam_matrix(idxAr = idxAr, idxXe = idxXe)

# calculator to the atoms object
atoms.add_calculator(
    calculator_class = Calculator,
    calculator_kwargs = {
        "epsilon": epsilon_mat, "sigma" : sigma_mat, 
        "positive_atom_idx" : idxXe, "negative_atom_idx" : idxAr,
        "mu0" : red.mu0, "d" : red.d0, "a" : red.a, 'd7':red.d7
    })

atoms.update_distance()

            ##########################
            ##########################
            ### INITIATE THE FIELD ###
            ##########################
            ##########################

k_vector_int = np.array(
    [[i,0,0] for i in range(1,150)] #+ [[0,i,0] for i in range(1,100)]
    ,dtype=np.float64)

print(k_vector_int.shape)

np.random.seed(2024)

amplitude = np.vstack([
    np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j
    for i in range(len(k_vector_int))
    ]) * 0e-1 * np.sqrt(L**3)

##################################
### FREE FIELD POTENTIAL BEGIN ###
##################################
Afield = CavityVectorPotential(
        k_vector_int = k_vector_int, amplitude = amplitude,
        Lxy = L, Lz = Lz, constant_c = red.c, coupling_strength = L * 1e1
        )
### FREE FIELD POTENTIAL END ###

            ############################
            ############################
            ### START THE SIMULATION ###
            ############################
            ############################

tlist = [t]
kinetic_elist = [atoms.kinetic()] 
potential_elist = [atoms.potential()] 
field_elist = [Afield.hamiltonian(return_sum_only=True)]

atoms.record(t)
Afield.record(t)

for i in tqdm(range(100000)):
    em_force_func = lambda t, atoms: Afield.force(t,atoms)

    atoms.Verlet_update(
            h = h, t = t,
            field_force = em_force_func
            )

    C_dot_tp1 = Afield.dot_amplitude(t+h,atoms)
    C_new = Afield.C + h * (C_dot_tp1)

    Afield.update_amplitude(C_new)
        
    t += h

    if i % 10 == 0:

        tlist.append(t)
        kinetic_elist.append(atoms.kinetic())
        potential_elist.append(atoms.potential())
        field_elist.append(Afield.hamiltonian(return_sum_only=True))

        atoms.record(t)
        Afield.record(t)

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

fig.savefig("figure/full_simulation.jpeg",dpi = 600)

result_dict = {
    'atoms': atoms,
    'field': Afield,
    'h' : h
}

with open('pickle_jar/single/log.pkl','wb') as handle:
    pickle.dump(result_dict, handle) 

