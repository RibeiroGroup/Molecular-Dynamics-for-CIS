import numpy as np

from distance import DistanceCalculator


class Atoms:
    """
    Class for collections of atoms
    Args:
    + elements (list of String): elements of the atoms in collections
        LEN: N for N is the number of atoms
    + R (np.array): list of positions
        SIZE: N same as N defined above
    + R_dot (np.array): list of velocity
        SIZE: N same as N defined above
    """
    def __init__(self, elements, R, R_dot):

        self.N_atoms = len(elements)
        assert isinstance(elements, list) 
        assert R.shape == self.N_atoms
        assert R_dot.shape == self.N_atoms

        self.elements = elements
        self.R = R
        self.R_dot = R_dot

