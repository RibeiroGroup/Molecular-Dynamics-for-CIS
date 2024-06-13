import numpy as np

#from distance import DistanceCalculator

test = True

class AtomsInBox:
    """
    Class for collections of atoms
    """
    def __init__(self, box_length):
        """
        Args:
        + box_length (float): the length of the box that contains atoms
        """

        self.N_atoms = 0

        self.elements = []

        self.R = None
        self.R_dot = None

        self.L = box_length

    def add(self, elements, R, R_dot):
        """
        Adding atoms to the "box"
        Args:
        + elements (list of String): elements of the atoms in collections
            LEN: N for N is the number of atoms
        + R (np.array): list of positions, no positions larger than the box length
            SIZE: N same as N defined above
        + R_dot (np.array): list of velocity
            SIZE: N same as N defined above
        """

        assert isinstance(elements, list) 

        for element in elements:
            assert element == "Ar" or element == "Xe"

        self.elements += elements
        self.N_atoms += len(elements)

        assert R.shape[0] == len(elements)
        assert not np.any(R > self.L)
        assert R_dot.shape[0] == len(elements)
        
        if self.R is None and self.R_dot is None:
            self.R = R
            self.R_dot = R_dot
        else:
            self.R = np.hstack([self.R, R])
            self.R_dot = np.hstack([self.R_dot, R_dot])

    def random_initialize(self, atoms, max_velocity, min_velocity = 0):
        """
        Args:
        + atoms (dict): python dictionary with keys are elements and 
            values are position
        """

        elements = []
        total_natoms = 0
        for el, n_atoms in atoms.items():
            elements += [el] * n_atoms
            total_natoms += n_atoms

        R = np.random.uniform(
                low = 0, high = self.L, size = (total_natoms, 3))

        R_dot = np.random.uniform(
                low = min_velocity, high = max_velocity,
                size = (total_natoms, 3))

        #calculate the magnitude of the velocity
        V = np.sqrt(np.einsum("ni,ni->n",R_dot,R_dot)) 
        #scaling the veclocity so that all veclocity magnitude is below the maximum
        scaler = np.where(max_velocity / V > 1 , 1, max_velocity / V)
        #if V > max_velocity, it will be scaled by -^
        scaler = np.tile(scaler[:,np.newaxis],(1,3))

        R_dot *= scaler

        self.add(elements, R, R_dot)

if test == True:
    atoms = AtomsInBox(box_length = 100)

    atoms.random_initialize({"Ar":50,"Xe":50}, max_velocity = 10)

