import numpy as np
from numba import jit
from utils import orthogonalize, timeit

test = True

class FreeFieldVectorPotential:
	"""
	Class for vector potential of the electromagnetic field in free (e.g. not confined) field
	Args:
	+ k_vector (np.array): wavevector, SIZE: N x 3 with N is arbitrary number of modes
	+ amplitude (np.array): amplitude of the vector potential for each mode, thus, it should
		have SIZE: N x 2 with N is the number of modes and should be consistent with the number
		of wavevector
	+ V (float): volume
	+ epsilon (np.array, optional): polarization vector, SIZE N x 2 x 3 with N should be consistent
		with above arguments
	"""
	def __init__(self, k_vector, amplitude, V, epsilon = None):
		self.number_modes = k_vector.shape[0]

		assert k_vector.shape[1] == 3
		assert amplitude.shape == (self.number_modes, 2)

		self.k_vector = k_vector
		self.C = amplitude
		self.V = V

		if epsilon == None:
			self.epsilon = []

			for k_vec in self.k_vector:
				self.epsilon.append(orthogonalize(k_vec)[1:,:])

			self.epsilon = np.array(self.epsilon)

		else:
			self.epsilon = epsilon

		assert self.epsilon.shape == (self.number_modes, 2, 3)

		self.omega = np.einsum("ni,ni->n",self.k_vector,self.k_vector)

	@timeit
	def __call__(self, t, R):
		"""
		Evaluate the vector potential at time t and multiple positions specified in R
		Args:
		+ t (float): time
		+ R (np.array): position. SIZE: M x 3 with M is the number of positions
		Returns:
		+ np.array: SIZE M x 3 with M specified in R argument
		"""

		C = self.C
		k_vec = self.k_vector
		pol_vec = self.epsilon

		omega = np.tile(self.omega[:,np.newaxis], (1, R.shape[0]))

		# free field mode function and c.c., a.k.a. exp(ikr) exp(-i \omega t) + c.c
		f_R = np.exp(
			1j * np.einsum("kj,mj->km",k_vec,R) - 1j * omega * t)

		fs_R = np.exp(
			-1j * np.einsum("kj,mj->km",k_vec,R) + 1j * omega * t)

		#Multiply C and epsilon_k (pol_vec), the outcome shape is N_modes x 3
		# (sum over dim 2, which is the number of polarized vector)
		C_epsilon_k = np.einsum("kj,kji->ki" ,C ,pol_vec)

		#
		A_R = np.einsum("ki,km->mi",C_epsilon_k, f_R) \
			+ np.einsum("ki,km->mi",np.conjugate(C_epsilon_k), fs_R)

		return A_R

	def dot(self, R):
		"""
		Calculate the time derivative of the vector potential
			
		"""
		pass

	def gradient(self, R):
		pass

	def get_electric_field(self, R):
		pass

	def get_magnetic_field(self, R):
		pass



if test:
	from utils import EM_mode_generate

	k_vec = EM_mode_generate(20)
	print(k_vec.shape)

	n_mode = k_vec.shape[0]

	amplitude = np.random.rand(n_mode,2) + 1j * np.random.rand(n_mode,2)

	A = FreeFieldVectorPotential(k_vector = k_vec, amplitude = amplitude, V = 100)

	R = np.random.rand(1000,3)

	A(0, R)






