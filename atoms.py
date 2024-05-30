import numpy as np

from utils import PBC_wrapping, orthogonalize, EM_mode_generate

from distance import DistanceCalculator, explicit_test
from neighborlist import neighbor_list_mask

from forcefield import LennardJonesPotential, explicit_test_LJ
from dipole import SimpleDipoleFunction

from reduced_parameter import sigma_ as len_unit, epsilon_ as energy_unit, time_unit, \
    M, c as v_light

from reduced_parameter import epsilon_Ar_Ar, epsilon_Xe_Xe, epsilon_Ar_Xe, sigma_Ar_Ar, sigma_Xe_Xe, \
    sigma_Ar_Xe, M_Ar, M_Xe, mu0, d0, a

class Atoms:
	def __init__(
		self, elements, position, velocity,
		forcefield, dipole_function,
		pbc_size, neighborlist_cutoff, 
		):

		self.elements = elements		
		self.r = position
		self.v = velocity
		
		self.L = pbc_size
		self.cell_width = neighborlist_cutoff

		self.forcefield = forcefield
		neighborlist_mask = neighbor_list_mask(
			r, self.L, self.cell_width)

	def force(self):
		pass

	def energy(self):
		pass

	def dipole(self):
		pass

	def velocity_step(self, h):
		pass

	def position_step(self, h):
		pass

