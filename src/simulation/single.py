import numpy as np

"""
    sample = sampler()
    r_ar, r_xe = sample["r"]
    r_dot_ar, r_dot_xe = sample["r_dot"]

    atoms = initiate_atoms_box()
    atoms.add(elements = ["Ar"]*N_atom_pairs,r = r_ar,r_dot = r_dot_ar)
    atoms.add(elements = ["Xe"]*N_atom_pairs,r = r_xe,r_dot = r_dot_xe)

    atoms.update_distance()
"""

def single_collision_simulation(
        cycle_number, t0, h, atoms, probe_field,
        cavity_field = None, total_dipole_threshold = 1e-5, 
        min_steps = 100, max_steps = 10000
        ):

    t = t0
    atoms.record(t)

    if probe_field:
        probe_field.record(t)

    if cavity_field:
        cavity_field.record(t)

    dipole_drop_flag = False
    potential_drop_flag = False
    steps = 0

    while (not dipole_drop_flag or abs(dipole) > total_dipole_threshold or steps < min_steps)\
            and steps < max_steps:
        steps += 1

        if cavity_field is not None and probe_field is not None:
            em_force_func = lambda t, atoms: \
                cavity_field.force(t,atoms) + probe_field.force(t,atoms)
        elif probe_field is not None:
            em_force_func = lambda t, atoms: \
                probe_field.force(t,atoms)
        else: em_force_func = None

        atoms.record(t)
        atoms.Verlet_update(
                h = h, t = t,
                field_force = em_force_func
                )
        
        if probe_field:
            probe_field.record(t)
            C_dot_tp1 = probe_field.dot_amplitude(t+h,atoms)
            C_new = probe_field.C + h * (C_dot_tp1)

            probe_field.update_amplitude(C_new)
        else: probe_field = None

        if cavity_field:
            cavity_field.record(t)
            C_dot_tp1 = cavity_field.dot_amplitude(t+h,atoms)
            C_new = cavity_field.C + h * (C_dot_tp1)

            cavity_field.update_amplitude(C_new)
        else: cavity_field = None

        t += h

        dipole = atoms.observable["total_dipole"][-1]
        potential = atoms.observable["potential"][-1]

        print(cycle_number,"\t",dipole, "\t", potential)

        if dipole < atoms.observable["total_dipole"][-2]:
            dipole_drop_flag = True
        elif dipole > atoms.observable["total_dipole"][-2]:
            dipole_drop_flag = False

    result = {
            "atoms":atoms, "cavity_field":cavity_field, "probe_field":probe_field
            }

    return t, result

