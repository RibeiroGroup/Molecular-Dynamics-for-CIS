import numpy as np

from calculator import Calculator
from utils import PBC_wrapping, neighborlist_mask

test = False

class AtomsInBox:
    """
    Class for collections of atoms
    """
    def __init__(self, box_length, cell_width, mass_dict):
        """
        Args:
        + box_length (float): the length of the box that contains atoms
        """
        
        self.L = box_length
        self.cell_width = cell_width

        self.calculator = None

        self.N_atoms = 0
        self.mass_dict = mass_dict

    def add(self, elements, R, R_dot):
        """
        Adding atoms to the "box"
        Args:
        + elements (list of String): elements of the atoms in collections
            LEN: N for N is the number of atoms
        + R (np.array): list of positions, no positions larger than the box length
            SIZE: N x 3 same as N defined above
        + R_dot (np.array): list of velocity
            SIZE: N x 3 same as N defined above
        """

        assert isinstance(elements, list) 

        for element in elements:
            assert element == "Ar" or element == "Xe"

        assert R.shape == (len(elements), 3)
        assert not np.any(R > self.L)
        assert R_dot.shape == (len(elements), 3)

        mass = np.array(
                list(map(lambda e: self.mass_dict[e], elements)
                    ))
        
        try:
            self.R = np.vstack([self.R, R])
            self.R_dot = np.vstack([self.R_dot, R_dot])
            self.elements += elements
            self.N_atoms += len(elements)
            self.mass += mass
        except AttributeError:
            self.R = R
            self.R_dot = R_dot
            self.elements = elements
            self.N_atoms = len(elements)
            self.mass = mass

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

    def update(self, R, R_dot, update_distance = True):

        self.R = PBC_wrapping(R,self.L)
        self.R_dot = R_dot

        if update_distance:

            neighborlist = neighborlist_mask(self.R, L = self.L, cell_width = self.cell_width)

            self.calculator.calculate_distance(self.R, neighborlist)

    def element_idx(self,element):

        atom_idx = np.array(
                list(map( lambda e: True if e == element else False, self.elements)))

        return atom_idx

    def add_calculator(self, calculator_kwargs, calculator_class=Calculator):

        self.calculator = calculator_class(
                N = self.N_atoms, box_length =  self.L, **calculator_kwargs)

        neighborlist = neighborlist_mask(self.R, L = self.L, cell_width = self.cell_width)

        self.calculator.calculate_distance(self.R, neighborlist)

    def acceleration(self, other_force_func = None):
        force = self.calculator.force()

        if other_force_func is not None:
            force += other_force_func(self)

        a = force / np.tile(self.mass[:,np.newaxis], (1,3))

        return a

    def Verlet_update(self, h):
        a = self.acceleration()

        v_half = self.R_dot + h * a / 2
        r_new = self.R + h * v_half

        self.update(R = r_new, R_dot = v_half)

        a = self.acceleration()
        v_new = v_half + h * a / 2

        self.update(R = r_new, R_dot = v_new, update_distance = False)
    
    def potential(self):
        return self.calculator.potential()

    def dipole(self,return_matrix = False):
        return self.calculator.dipole(return_matrix = False)

    def dipole_grad(self):
        return self.calculator.dipole_grad()

    def kinetic_energy(self):
        k = 0.5 * self.mass * np.einsum("ni,ni->n",self.R_dot,self.R_dot)
        return np.sum(k)

if test == True:
    import reduced_parameter as red

    atoms = AtomsInBox(box_length = 20, cell_width = 5, mass_dict = red.mass_dict)

    atoms.random_initialize({"Ar":5,"Xe":5}, max_velocity = 10)

    idxAr = atoms.element_idx(element = "Xe")
    idxXe = atoms.element_idx(element = "Ar")

    epsilon_mat, sigma_mat = red.generate_LJparam_matrix(idxAr = idxAr, idxXe = idxXe)

    atoms.add_calculator(calculator_kwargs = {
        "epsilon": epsilon_mat, "sigma" : sigma_mat, 
        "positive_atom_idx" : idxXe, "negative_atom_idx" : idxAr,
        "mu0" : red.mu0, "d" : red.d0, "a" : red.a
        })

    gradD = atoms.dipole_grad()
    print(gradD.shape)

