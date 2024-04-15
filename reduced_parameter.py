import numpy as np
from scipy.constants import physical_constants, epsilon_0, Avogadro, e as e_charge

###############
### LJ unit ###
###############

sigma = 3.41 #unit: Angstrom
epsilon = 0.996 #unit: kJ/mol
M = 39.948 #a.u. or g/mol

##########
### SI ###
##########

sigma_   = sigma    * 1e-10             #convert to m
epsilon_ = epsilon  * 1e3  /Avogadro    #convert to J
M_       = M        * 1e-3 /Avogadro    #convert to kg

###############################
### ASSOCIATED REDUCED UNIT ###
###############################

dipole_unit = np.sqrt(4 * np.pi * epsilon_0 * epsilon_ * sigma_**3)

########################################
### Calculating reduced LJ parameter ###
########################################

sigma_Ar_Ar = 3.410 / sigma 
sigma_Ar_Xe = 3.735 / sigma
sigma_Xe_Xe = 4.060 / sigma

epsilon_Ar_Ar = 0.996 / epsilon
epsilon_Ar_Xe = 1.377 / epsilon
epsilon_Xe_Xe = 1.904 / epsilon

M_Ar = 39.948 / M
M_Xe = 131.293 / M

############################################
### Calculating reduced dipole parameter ###
############################################

bohr_rad, _, _ = physical_constants["Bohr radius"]

mu0_1 = (0.0124 * e_charge * 1e-10) / dipole_unit

a1 = 1.5121 * sigma_ / bohr_rad

d0_1 = 7.10 * bohr_rad / sigma_








