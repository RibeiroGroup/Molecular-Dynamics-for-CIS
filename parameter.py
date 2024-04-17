import constants

from scipy.constants import physical_constants, Avogadro
from scipy.constants import epsilon_0, speed_of_light, proton_mass, neutron_mass, electron_mass

test = 0

epsilon_Ar_Ar = 0.996 * 1.59360e-3
epsilon_Ar_Xe = 1.377 * 1.59360e-3
epsilon_Xe_Xe = 1.904 * 1.59360e-3

sigma_Ar_Ar = 3.41 * (1e-10 / 5.29177e-11)
sigma_Ar_Xe = 3.735* (1e-10 / 5.29177e-11)
sigma_Xe_Xe = 4.06 * (1e-10 / 5.29177e-11)

M_Ar = (18 * proton_mass + (40 - 18) * neutron_mass) / electron_mass
M_Xe = (54 * proton_mass + (131 - 54) * neutron_mass)/ electron_mass 

# parameters for Grigorieve dipole function
mu0 = 0.0124
a = 1.5121
d0 = 7.10

bohr_rad, _, _ = physical_constants["Bohr radius"]

if test:
    argon_density = 1.784 * 1e-3 #(g/m^3)
    argon_density = argon_density * Avogadro / 39.948 #molecules / m^3
    print(argon_density)
    argon_density = argon_density * (bohr_rad**3)
    print(argon_density)
