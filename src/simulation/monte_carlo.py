import time
import pickle
import os, sys

import numpy as np

class MonteCarloSim:
    def __init__()

    for i in range(config.num_cycles):
        sample = sampler()
        r_ar, r_xe = sample["r"]
        r_dot_ar, r_dot_xe = sample["r_dot"]

        atoms = initiate_atoms_box()
        atoms.add(elements = ["Ar"]*N_atom_pairs,r = r_ar,r_dot = r_dot_ar)
        atoms.add(elements = ["Xe"]*N_atom_pairs,r = r_xe,r_dot = r_dot_xe)

        atoms.update_distance()

        atoms.record(t)
        cavity_field.record(t)
        probe_field.record(t)

        dipole_drop_flag = False
        potential_drop_flag = False
        steps = 0

        while (not dipole_drop_flag or abs(dipole) > 1e-5 or steps < 100) and steps < 10000:
            steps += 1

            em_force_func = lambda t, atoms: \
                cavity_field.force(t,atoms) + probe_field.force(t,atoms)

            atoms.Verlet_update(
                    h = h, t = t,
                    field_force = em_force_func
                    )

            C_dot_tp1 = cavity_field.dot_amplitude(t+h,atoms)
            C_new = cavity_field.C + h * (C_dot_tp1)

            cavity_field.update_amplitude(C_new)
                
            C_dot_tp1 = probe_field.dot_amplitude(t+h,atoms)
            C_new = probe_field.C + h * (C_dot_tp1)

            probe_field.update_amplitude(C_new)
            t += h

            atoms.record(t)
            cavity_field.record(t)
            probe_field.record(t)

            dipole = atoms.observable["total_dipole"][-1]
            potential = atoms.observable["potential"][-1]

            print(i,"\t",dipole, "\t", potential)

            if dipole < atoms.observable["total_dipole"][-2]:
                dipole_drop_flag = True
            elif dipole > atoms.observable["total_dipole"][-2]:
                dipole_drop_flag = False

        result = {
                "atoms":atoms, "cavity_field":cavity_field, "probe_field":probe_field,
                "temperature":K_temp, "mu0" : config.mu0, "seed":[config.seed1, config.seed2]
                }
        with open("pickle_jar/result_cavity_{}.pkl".format(i),"wb") as handle:
            pickle.dump(result, handle)

        del atoms
        new_cavity_field = CavityVectorPotential(
            kappa = kappa, m = m, L = L, S = L ** 2,
            amplitude = cavity_field.C,
            constant_c = red.c,
            )

        new_probe_field = FreeVectorPotential(
                k_vector = config.probe_kvector, 
                amplitude = probe_field.C,
                V = L ** 3, constant_c = red.c,
                )

        del probe_field
        probe_field = new_probe_field

        del cavity_field
        cavity_field = new_cavity_field
