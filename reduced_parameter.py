import numpy as np
import scipy.constants as scicon
from scipy.constants import physical_constants, Avogadro, e as e_charge
import constants

test = 1

###############
### LJ unit ###
###############

sigma_ = 3.41       #unit: Angstrom
epsilon_ = 0.996    #unit: kJ/mol
M_ = 39.948         #unit: a.u. or g/mol
k_e = 1 / (4 * np.pi * scicon.epsilon_0)

##########
### SI ###
##########

sigma   = sigma_    * 1e-10             #convert to m
epsilon = epsilon_  * 1e3  /Avogadro    #convert to J
M       = M_        * 1e-3 /Avogadro    #convert to kg

###############################
### ASSOCIATED REDUCED UNIT ###
###############################

dipole_unit = np.sqrt(4 * np.pi * scicon.epsilon_0 * epsilon * sigma**3)
time_unit = np.sqrt(M * sigma**2/epsilon)

########################################
### Calculating reduced LJ parameter ###
########################################

sigma_Ar_Ar = 3.410 / sigma_
sigma_Ar_Xe = 3.735 / sigma_
sigma_Xe_Xe = 4.060 / sigma_

epsilon_Ar_Ar = 0.996 / epsilon_
epsilon_Ar_Xe = 1.377 / epsilon_
epsilon_Xe_Xe = 1.904 / epsilon_

M_Ar = 39.948 / M_
M_Xe = 131.293 / M_

############################################
### Calculating reduced dipole parameter ###
############################################

bohr_rad, _, _ = physical_constants["Bohr radius"]

mu0 = (0.0124 * e_charge * 1e-10) / dipole_unit

a = 1.5121 * sigma / bohr_rad

d0 = 7.10 * bohr_rad / sigma

d = 10

#####################################
### Calculating reduced constants ###
#####################################

epsilon_0 = scicon.epsilon_0 * k_e
c = 3e8 * np.sqrt(M / epsilon)


if test:
    print("Epsilon (kJ/mol)", epsilon_)
    print("Reduced epsilon (Ar-Ar, Ar-Xe, Xe-Xe):", 
            epsilon_Ar_Ar,";", epsilon_Ar_Xe,";", epsilon_Xe_Xe)

    print("######################")
    print("Sigma (Angstrom)", sigma_)
    print("Reduced sigma (Ar-Ar, Ar-Xe, Xe-Xe):", 
            sigma_Ar_Ar,";", sigma_Ar_Xe,";", sigma_Xe_Xe)

    print("######################")
    print("Mass (kg/mol)",M_)
    print("Reduced Mass (Ar, Xe):", M_Ar,";",M_Xe)

    print("######################")
    print("Dipole unit multiple (C . m)", dipole_unit)
    print("Reduced dipole parameter:")
    print("mu0 = ",mu0)
    print("a = ",a)
    print("d0 = ",d0)

    print("######################")
    print("Time", time_unit)

    print("######################")
    print("epsilon_0: ", epsilon_0)
    print("c: ", c)

