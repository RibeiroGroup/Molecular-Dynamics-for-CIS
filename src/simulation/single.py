import numpy as np

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
    + atoms (Atoms object): atoms
    + field (BaseVectorPotential or inherited-class objects): field
    + t0 (float): initial time, just for printing
    + potential_threshold (float): termination conditions, if the system total 
        potential value fall below this value, the patient (see patient argument) value 
        will be counted down, and the simulation is terminated if patient drop to zero
    + min_steps (int): minimum number of steps, the simulation is guaranteed to not 
        stop if number of steps < min_steps
    + max_steps (int): maximum number of steps, the simulation stop if number of steps 
        exceed max_steps
    + patient (int): number of steps to count down if the terminated condition met,
        the count down is reset if the termination condition is not satisfied
    + record_every (int): record every N steps with N is the provided value.
        Example: N = 10, the data is record at 10, 20, 30, ... , 100, ...-th step
    """

    t = t0
    # record the data in atoms object
    atoms.record(t)

    if field:
        # record the data in the field object
        field.record(t)

    steps = 0
    patient_steps = 0

    # condition for continuing the loop: 
    while (steps < min_steps or abs(potential) > potential_threshold)\
            and steps < max_steps:
        steps += 1

        if field is not None:
            # the force exerted by the field will the 
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

