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
        cycle_number, h, atoms, t0 = 0,
        field = None, potential_threshold = 1e-5, 
        min_steps = 200, max_steps = 10000,
        patient = 50, record_every = 1
        ):
    """
    Propagate one system of Ar-Xe atoms and EM field
    Args:
    + cycle_number (int): just for printing purpose
    + h (float): time step
    + atoms (
    + t0 (float): initial time
    """

    t = t0
    atoms.record(t)

    if field:
        field.record(t)

    steps = 0
    patient_steps = 0

    while (steps < min_steps or abs(potential) > potential_threshold)\
            and steps < max_steps:
        steps += 1

        if field is not None:
            em_force_func = lambda t, atoms: field.force(t,atoms)

        else: em_force_func = None

        atoms.Verlet_update(
                h = h, t = t,
                field_force = em_force_func
                )
        
        if field:
            C_dot_tp1 = field.dot_amplitude(t+h,atoms)
            C_new = field.C + h * (C_dot_tp1)

            field.update_amplitude(C_new)
        else: field = None

        t += h

        if steps % record_every == 0:

            atoms.record(t)
            if field: field.record(t)

            dipole = atoms.observable["total_dipole"][-1]
            potential = atoms.observable["potential"][-1]

            print("Cycle: {}, iterations: {}, total dipole: {:.4E}, potential: {:.4E}".format(
                cycle_number,steps,dipole, potential))

            if abs(potential) < potential_threshold:
                patient_steps += 1
            else: 
                patient_steps = 0

    result = {
            "atoms":atoms, "field":field
            }

    return t, result

