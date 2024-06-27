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

    def add(self, elements, r, r_dot):
        """
        Adding atoms to the "box"
        Args:
        + elements (list of String): elements of the atoms in collections
            LEN: N for N is the number of atoms
        + r (np.array): list of positions, no positions larger than the box length
            SIZE: N x 3 same as N defined above
        + r_dot (np.array): list of velocity
            SIZE: N x 3 same as N defined above
        """

        assert isinstance(elements, list) 

        for element in elements:
            assert element == "Ar" or element == "Xe"

        assert r.shape == (len(elements), 3)
        assert not np.any(r > self.L)
        assert r_dot.shape == (len(elements), 3)

        mass = np.array(
                list(map(lambda e: self.mass_dict[e], elements)
                    ))
        
        try:
            self.r = np.vstack([self.r, r])
            self.r_dot = np.vstack([self.r_dot, r_dot])
            self.elements += elements
            self.N_atoms += len(elements)
            self.mass += mass
        except AttributeError:
            self.r = r
            self.r_dot = r_dot
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

        r = np.random.uniform(
                low = 0, high = self.L, size = (total_natoms, 3))

        r_dot = np.random.uniform(
                low = min_velocity, high = max_velocity,
                size = (total_natoms, 3))

        #calculate the magnitude of the velocity
        V = np.sqrt(np.einsum("ni,ni->n",r_dot,r_dot)) 
        #scaling the veclocity so that all veclocity magnitude is below the maximum
        scaler = np.where(max_velocity / V > 1 , 1, max_velocity / V)
        #if V > max_velocity, it will be scaled by -^
        scaler = np.tile(scaler[:,np.newaxis],(1,3))

        r_dot *= scaler

        self.add(elements, r, r_dot)

    def update(self, r, r_dot, update_distance = True):

        self.r = PBC_wrapping(r,self.L)
        self.r_dot = r_dot

        if update_distance:

            neighborlist = neighborlist_mask(self.r, L = self.L, cell_width = self.cell_width)

            self.calculator.calculate_distance(self.r, neighborlist)

    def element_idx(self,element):

        atom_idx = np.array(
                list(map( lambda e: True if e == element else False, self.elements)))

        return atom_idx

    def add_calculator(self, calculator_kwargs, calculator_class=Calculator):

        self.calculator = calculator_class(
                N = self.N_atoms, box_length =  self.L, **calculator_kwargs)

        neighborlist = neighborlist_mask(self.r, L = self.L, cell_width = self.cell_width)

        self.calculator.calculate_distance(self.r, neighborlist)

    def acceleration(self, other_force_func = None):
        force = self.calculator.force()

        if other_force_func is not None:
            force += np.real(other_force_func(self))

        a = force / np.tile(self.mass[:,np.newaxis], (1,3))

        return a

    def Verlet_update(self, h, other_force_func = None):
        a = self.acceleration(other_force_func)

        v_half = self.r_dot + h * a / 2
        r_new = self.r + h * v_half

        self.update(r = r_new, r_dot = v_half)

        a = self.acceleration(other_force_func)
        v_new = v_half + h * a / 2

        self.update(r = r_new, r_dot = v_new, update_distance = False)
    
    def potential(self):
        return self.calculator.potential()

    def dipole(self,return_matrix = False):
        return self.calculator.dipole(return_matrix = False)

    def charge(self):
        return self.calculator.dipole_grad()

    def kinetic(self):
        k = 0.5 * self.mass * np.einsum("ni,ni->n",self.r_dot,self.r_dot)
        return np.sum(k)

    def current_mode_projection(self):
        q = self.charge()
        return np.einsum("nij,ni->nj",q,self.r_dot)


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

