import numpy as np
import utilities.reduced_parameter as red

#def get_full_dipole():

def get_colliding_time(
        atoms, dipole_threshold, convert_time = True, mu0 = red.mu0, 
        ):
    """
    Calculating the collision time for each pairs of colliding Argon-Xenon
    Warning: this code is mean to used with result of simulation_monte.py code,
    the i-th element (i < N/2) of atoms.r is supposed to be of Xenons, and the last 
    N_pair of atoms.r is supposed to be of Argon, thus the pair-wise dipole can be 
    computed
    Args:
    + atoms (Atoms object): Atoms
    + dipole_threshold (float): dipole value at which the collision is assumed to
        happen and end
    + convert_time (bool, default = True):
    + mu0 (float, deffault = True):
    Output:
    + 
    """

    #calculate the duration of the collision
    time = red.convert_time(
            np.array(atoms.trajectory["t"]))

    N_pairs = int(atoms.N_atoms / 2)
    traj_len = len(atoms.trajectory["r"])

    # calculating list of LIST of dipole for each in N_pairs of Argon-Xenon
    dipole_vs_time = []
    dlist = []

    for i in range(traj_len):
        r = atoms.trajectory["r"][i]

        # Note that simulation run from simulation_monte.py arrange argons' and xenon' indices separately
        # e.g. R = [r_{Ar,1}, r_{Ar,2}, ... , r_{Ar,N}, r_{Xe,1}, r_{Xe,2}, ... , r_{Xe,N}]
        r_ar = r[0:N_pairs]
        r_xe = r[N_pairs:]
        dvec = (r_ar - r_xe)

        d = np.sqrt(np.einsum("ni,ni->n",dvec,dvec))
        dlist.append(d)

        dipole = mu0 * np.exp(-red.a * (d - red.d0)) - red.d7/d**7

        dipole_vs_time.append(dipole)

    dipole_vs_time = np.array(dipole_vs_time)
    dlist = np.array(dlist)

    colliding_time = []
    for i in range(dipole_vs_time.shape[1]):

        t = time[dipole_vs_time[:,i] > dipole_threshold]

        if len(t) > 0:
            dt = t[-1] - t[0]
        else: dt = 0

        colliding_time.append(dt)

    return colliding_time
