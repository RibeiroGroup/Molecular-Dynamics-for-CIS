import argparse
import numpy as np
import sympy as sm

from utils import PBC_wrapping

# Set run_test to 1 to run test

def generate_dipole_mask(positive_atom_idx, negative_atom_idx, all_distance = False):
    """
    Generating dipole mask (potentially for Calculator object's dipole method)
    Output will be a N x N mask with True at ij element if there is dipole between
    them, e.g. positive and negative atoms, and False otherwise.
    Args:
    + positive_atom_idx (np.array): array with SIZE: N (N is number of atoms)
        and element is either 1/True or 0/False
    + negative_atom_idx (np.array): array with SIZE: N (N is number of atoms)
        and element is either 1/True or 0/False
    + all_distance (bool): if True, will return a matrix where ALL ij position 
        with dipoles can be form btw i-th atom and j-th atom are marked as True
        If False, only position that correspond to distance vector r_positive - r_negative
        marked as True. The second option for correctly compute the sign of dipole
    """

    r_pn = np.array(
        np.outer(positive_atom_idx, negative_atom_idx), dtype = bool)

    if all_distance:
        r_pn += r_pn.transpose()

    return r_pn

def explicit_test_dipole(R, dipole_mask, mu0, a, d, L):

    dipole = np.zeros((len(R),len(R), 3))
    dipole_grad = np.zeros((len(R),len(R), 3, 3))

    for i, ri in enumerate(R):
        for j, rj in enumerate(R):
            if i == j : continue
            if not dipole_mask[i,j]: continue

            #calculating distance
            distance_vec = PBC_wrapping(ri - rj, L)
            distance = np.sqrt(distance_vec @ distance_vec)

            #calculating dipole vector
            dipole_ = mu0 * np.exp(-a * (distance - d)) 
            dipole_ *= distance_vec
            dipole[i,j,:] = dipole_

            # Calculating dipole gradient tensor
            exp_ad = np.exp(-a * (distance - d))

            distance_outer = np.einsum("j,k->jk", distance_vec, distance_vec)

            gradient = - a * mu0 * distance_outer * exp_ad / distance**2
            gradient -= mu0 * distance_outer * exp_ad / distance ** 3
            gradient += (mu0 * exp_ad / distance) * np.eye(3)

            dipole_grad[i,j,:,:] = gradient

    dipole += 1 * np.swapaxes(dipole,0,1)
    dipole_grad += -1 * np.swapaxes(dipole_grad,0,1)

    return dipole, dipole_grad

def Grigoriev_dipole(distance, distance_vec, mu0, a, d):
    dipole = mu0 * np.exp(-a * (distance - d)) 

    dipole = np.tile(
           dipole[:,np.newaxis], (1,3))

    # multiplying with distance vec to generate distance vector
    dipole *= distance_vec

    return dipole

def Grigoriev_dipole_grad(distance, distance_vec, mu0, a, d):
    distance = np.tile(distance[:,np.newaxis,np.newaxis], (1,3,3))

    exp_ad = np.exp(-a * (distance - d))

    distance_outer = np.einsum("ij,ik->ijk", distance_vec, distance_vec)

    gradient = - a * mu0 * distance_outer * exp_ad / distance**2
    gradient -= mu0 * distance_outer * exp_ad / distance ** 3
    gradient += (mu0 * exp_ad / distance) * np.eye(3)

    return gradient

