from copy import deepcopy

import numpy as np

from .utils import PBC_wrapping, neighborlist_mask

class AtomsInBox:
    """
    Class for collections of atoms
    """
    def __init__(self, Lxy, Lz, mass_dict, cell_width = None):
        """
        Args:
        + box_length (float): the length of the box that contains atoms
        """
        
        self.Lxy = Lxy
        self.Lz = Lz

        assert isinstance(cell_width, tuple) and len(cell_width) == 2
        self.cell_width = cell_width

        self.calculator = None

        self.N_atoms = 0
        self.mass_dict = mass_dict

        self.N_atoms = 0
        self.r = None
        self.r_dot = None
        self.elements = None

        self.trajectory = {"t":[],"r":[],"r_dot":[]}
        self.observable = {
                "t":[],"kinetic":[],"potential":[],"total_dipole":[],
                'dipole':[], 'dipole_velocity': []
                }

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
        assert not np.any(r[:,:2] > self.Lxy) and not np.any(r[:,-1] > self.Lz)
        assert r_dot.shape == (len(elements), 3)

        mass = np.array(
                list(map(lambda e: self.mass_dict[e], elements)
                    ))

        if self.r is None:
            assert self.r_dot is None and self.elements is None
            self.r = r
            self.r_dot = r_dot
            self.elements = elements
            self.N_atoms = len(elements)
            self.mass = mass

        else:
            self.elements += elements
            self.N_atoms += len(elements)
            self.r = np.vstack([self.r, r])
            self.r_dot = np.vstack([self.r_dot, r_dot])
            self.mass = np.hstack([self.mass,mass])

    def record(self, t):
        """
        Add current position, velocity to trajectory dict and energy, dipole
        to observable dict
        Args:
        + t (float): time
        """
        self.trajectory["t"].append(t)
        self.trajectory["r"].append(deepcopy(self.r))
        self.trajectory["r_dot"].append(deepcopy(self.r_dot))
        
        self.observable["t"].append(t)
        self.observable["kinetic"].append(self.kinetic())
        self.observable["potential"].append(self.potential())

        self.observable["dipole_velocity"].append(
                self.current_mode_projection())
        self.observable["dipole"].append(self.dipole())

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

        r = np.hstack([
            np.random.uniform(
                low = 0, high = self.Lxy, size = (total_natoms, 2)),
            np.random.uniform(
                low = 0, high = self.Lz, size = (total_natoms, 1))
            ])

        r_dot = sample_velocity(total_natoms, max_velocity, min_velocity)

        self.add(elements, r, r_dot)

    def update_distance(self):

        assert self.N_atoms == self.calculator.N

        if self.cell_width is not None:
            cell_width_xy, cell_width_z = self.cell_width
            neighborlist = neighborlist_mask(
                    self.r, Lxy = self.Lxy, Lz = self.Lz, 
                    cell_width_xy = cell_width_xy, 
                    cell_width_z = cell_width_z)
            self.calculator.calculate_distance(self.r, neighborlist)
        else:
            self.calculator.calculate_distance(self.r)

    def update(self, r, r_dot, update_distance = True):
        
        assert self.N_atoms == self.calculator.N

        self.r = PBC_wrapping(r,Lxy=self.Lxy,Lz=self.Lz)
        self.r_dot = r_dot

        if update_distance:
            self.update_distance()

    def clear(self):
        self.N_atoms = 0
        self.r = None
        self.r_dot = None
        self.elements = None
        self.calculator.clear()

    def element_idx(self,element):

        atom_idx = np.array(
                list(map( lambda e: True if e == element else False, self.elements)))

        return atom_idx

    def add_calculator(self, calculator_class, calculator_kwargs, N_atoms = None):

        N_atoms = self.N_atoms if N_atoms is None else N_atoms

        self.calculator = calculator_class(
                N = N_atoms, Lxy = self.Lxy, Lz = self.Lz, **calculator_kwargs)

    def acceleration(self, t = None, field_force = None):
        force = self.calculator.force()

        if field_force is not None:
            assert t is not None
            force += np.real(field_force(t, self))

        a = force / np.tile(self.mass[:,np.newaxis], (1,3))

        return a

    def Verlet_update(self, h, t = None, field_force = None):
        a = self.acceleration(field_force=field_force, t=t)

        v_half = self.r_dot + h * a / 2
        r_new = self.r + h * v_half

        self.update(r = r_new, r_dot = v_half)

        a = self.acceleration(field_force=field_force, t=t+h)
        v_new = v_half + h * a / 2

        self.update(r = r_new, r_dot = v_new, update_distance = False)
    
    def potential(self):
        return self.calculator.potential()

    def dipole(self):
        return self.calculator.dipole(return_matrix = False)

    def total_dipole(self):
        dipole_vec = np.sum(
                self.calculator.dipole(return_matrix = False), axis = 0
                )
        return np.sqrt(np.sum(dipole_vec**2))

    def charge(self):
        return self.calculator.dipole_grad()

    def kinetic(self):
        k = 0.5 * self.mass * np.einsum("ni,ni->n",self.r_dot,self.r_dot)
        return np.sum(k)

    def current_mode_projection(self):
        q = self.charge()
        return np.einsum("nij,ni->nj",q,self.r_dot)


