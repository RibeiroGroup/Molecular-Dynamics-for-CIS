import numpy as np
import utilities.reduced_parameter as red

#def get_full_dipole():

def get_colliding_time(
        atoms, mu0, dipole_threshold, convert_time = True
        ):

    #calculate the duration of the collision
    time = red.convert_time(
            np.array(atoms.trajectory["t"]))

    N_pairs = int(atoms.N_atoms / 2)
    traj_len = len(atoms.trajectory["r"])
    dipole_vs_time = []
    for i in range(traj_len):
        r = atoms.trajectory["r"][i]

        r_ar = r[0:N_pairs]
        r_xe = r[N_pairs:]
        dvec = (r_ar - r_xe)

        d = np.sqrt(np.einsum("ni,ni->n",dvec,dvec))

        dipole = mu0 * np.exp(-red.a * (d - red.d0)) - red.d7/d**7

        dipole_vs_time.append(dipole)

    dipole_vs_time = np.array(dipole_vs_time)

    colliding_time = []
    for i in range(dipole_vs_time.shape[1]):

        t = time[dipole_vs_time[:,i] > dipole_threshold]

        if len(t) > 0:
            dt = t[-1] - t[0]
        else: dt = 0

        colliding_time.append(dt)

    return colliding_time
