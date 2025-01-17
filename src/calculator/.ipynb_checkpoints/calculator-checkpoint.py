import numpy as np
from .distance import DistanceCalculator
from .utils import PBC_wrapping

from .function import LJ_force, LJ_potential, \
    Grigoriev_dipole_, Grigoriev_dipole_grad_, generate_dipole_mask

class Calculator(DistanceCalculator):
    """
    Class for calculating various properties that based on interactomic distance.
    Args:
    + N (int): number of atoms
    + box_length (float): the length of the simulated box
    Follow is various properties parameters. Need to add/modify parameters if existing function 
    to be modified or new function to be added. Current function: Lennard-Jones potential
    and Griegoriev dipole function.
    - Lennard-Jones potential parameters:
        + epsilon (np.ndarray): epsilon of Lennard-Jones potential
            SIZE: N x N
        + sigma (np.ndarray): sigma of Lennard-Jones potential
            SIZE: N x N
    - Argon-Xenon dipole function:
        + positive_atom_idx (np.array): array with SIZE: N (N is number of atoms)
            and element is either 1/True or 0/False
        + negative_atom_idx (np.array): array with SIZE: N (N is number of atoms)
            and element is either 1/True or 0/False
        + mu0 (float): mu_0 parameter
        + a (float): a parameter
        + d(float): d parameter
    """
    def __init__(
        self, N, Lxy, Lz, epsilon, sigma, 
        positive_atom_idx, negative_atom_idx, mu0, a, d, d7
        ):

        super().__init__(N, Lxy=Lxy, Lz=Lz)

        assert epsilon.shape == (self.N, self.N)
        self.epsilon = epsilon

        assert sigma.shape == (self.N, self.N)
        self.sigma = sigma

        self.dipole_mask = generate_dipole_mask(
                positive_atom_idx, negative_atom_idx)

        self.mu0 = mu0
        self.a = a
        self.d = d
        self.d7 = d7

    def clear(self):
        self.distance_matrix = None
        self.distance_vec_tensor = None

    def calculate_distance(self, R, neighborlist = None, update_attr = True):
        """
        Calculating all relevant distances and distances vector.
        Note: Always running this method to update all distances matrix 
        before calculating dipole or force/ potential.
        Args:
        + R (np.array): list of all atoms' positions
            SIZE: N x N
        + neighborlist (np.array): neighborlist
            SIZE: N x N
        """

        # using methods from DistanceCalculator parent class to calculate 
        # distance matrix
        distance_matrix = self.calculate_distance_matrix(
            R, neighborlist)

        # and distance vector matrix
        distance_vec_tensor = self.calculate_distance_vector_tensor(
            R, neighborlist)

        if update_attr:
            self.distance_matrix = distance_matrix
            self.distance_vec_tensor = distance_vec_tensor

        # generating mask for subsequent calculation
        # usual upper triangle boolean matrix
        self.mask = self.utriang_mask

        if neighborlist is not None:
             self.mask *= neighborlist
             self.dipole_mask *= neighborlist

        self.mask_x3 = self.repeat_x3(self.mask)

        return distance_matrix, distance_vec_tensor

    def potential(self,return_matrix = False):
        """
        Method for calculating Lennard-Jones potential
        """

        # extracting epsilon and sigma in the parameter matrix
        # that correspond to the mask position
        epsilon = self.epsilon[self.mask]
        sigma = self.sigma[self.mask]

        # extracting distance in the distance matrix
        # that correspond to the mask position
        darray = self.distance_matrix[self.mask]

        # apply LJ_potential function
        potential = LJ_potential(sigma, epsilon, darray)
        
        if return_matrix:
            # rearrange the result array into matrix form
            return self.matrix_reconstruct(
                potential, symmetric_padding = 1, mask = self.mask)

        return np.sum(potential)

    def force(self, return_matrix = False):
        """
        Method for calculating Lennard-Jones force
        """
        # retrieving parameters that corresponding to relevant pair of atoms
        epsilon = self.epsilon[self.mask]
        sigma = self.sigma[self.mask]

        # retrieving distance value and vector similarly
        darray = self.distance_matrix[self.mask]
        dvec_array = self.distance_vec_tensor[self.mask_x3].reshape(-1,3)

        # calculating force
        force = LJ_force(sigma, epsilon, darray, dvec_array)
        force = self.matrix_reconstruct(
            force, symmetric_padding = -1, 
            mask = self.mask, utriang_mask_x3 = self.mask_x3)

        if not return_matrix: 
            force = np.sum(force, axis = 1)

        return force 

    def dipole(self, return_matrix=True):

        #generate the mask for retrieving relevant distance and parameters
        mask = self.dipole_mask
        mask_x3 = self.repeat_x3(mask)

        # generating distance array rather than matrix (similar for distance vec)
        distance = self.distance_matrix[mask]
        distance_vec = self.distance_vec_tensor[mask_x3].reshape(-1,3)

        dipole = Grigoriev_dipole_(
                distance = distance, distance_vec = distance_vec, 
                mu0 = self.mu0, a = self.a, d = self.d, d7 = self.d7)

        # rearrange the result array into N x N x 3 matrix form 
        dipole = self.matrix_reconstruct(
            dipole, symmetric_padding = 1, 
            mask = mask, utriang_mask_x3 = mask_x3)

        if not return_matrix: 
            dipole = np.sum(dipole, axis = 1)

        return dipole

    def dipole_grad(self, return_matrix = False):
        """
        Return gradient of the dipole function w.r.t. position r = r_+.
        The matrix will have the shape N x N x 3 x 3 where the 
        3 x 3 tensor at i, j -th element will be
            |~ dmu_x/dr_x dmu_y/dr_x dmu_z/dr_x ~|
            |  dmu_x/dr_y dmu_y/dr_y dmu_z/dr_y  |
            |_ dmu_x/dr_z dmu_y/dr_z dmu_z/dr_z _|; 
            OR (D_r mu)_(ij) = dmu_j / dr_i 
        with mu = mu(r_i, r_j), e.g. dipole vector between i-th and j-th atoms
        """

        mask = self.dipole_mask
        mask_x3 = self.repeat_x3(mask)

        distance = self.distance_matrix[mask]
        distance_vec = self.distance_vec_tensor[mask_x3].reshape(-1,3)

        gradient = Grigoriev_dipole_grad_(
            distance = distance, distance_vec = distance_vec,
            mu0 = self.mu0, a = self.a, d = self.d, d7 = self.d7)

        # rearrange the result array into N x N x 3 matrix form 
        gradient = self.matrix_reconstruct(
            gradient, symmetric_padding = -1, 
            mask = mask, utriang_mask_x3 = mask_x3)

        if not return_matrix:
            gradient = np.sum(gradient, axis = 1)

        return gradient


