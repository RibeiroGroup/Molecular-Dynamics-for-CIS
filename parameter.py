from scipy.constants import physical_constants
from scipy.constants import epsilon_0, speed_of_light, proton_mass, neutron_mass, electron_mass

epsilon_Ar_Ar = 0.996 * 1.59360e-3
epsilon_Ar_Xe = 1.377 * 1.59360e-3
epsilon_Xe_Xe = 1.904 * 1.59360e-3

sigma_Ar_Ar = 3.41 * (1e-10 / 5.29177e-11)
sigma_Ar_Xe = 3.735* (1e-10 / 5.29177e-11)
sigma_Xe_Xe = 4.06 * (1e-10 / 5.29177e-11)

M_Ar = (18 * proton_mass + (40 - 18) * neutron_mass) / electron_mass
M_Xe = (54 * proton_mass + (131 - 54) * neutron_mass)/ electron_mass 

# parameters for Grigorieve dipole function
mu0_1 = 0.0124
a1 = 1.5121
d0_1 = 7.10

