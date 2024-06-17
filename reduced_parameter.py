import numpy as np
import scipy.constants as scicon
from scipy.constants import physical_constants, Avogadro, e as e_charge
import constants

test = 0

###############
### LJ unit ###
###############

sigma_ = 3.41       #unit: Angstrom
epsilon_ = 0.996    #unit: kJ/mol
M_ = 39.948         #unit: a.u. or g/mol

###########
### CGS ###
###########

sigma   = sigma_    * 1e-8                   #convert to cm
epsilon = epsilon_  * 1e3 * 1e7  /Avogadro   #convert to erg
M       = M_ / Avogadro                      #convert to g

###############################
### ASSOCIATED REDUCED UNIT ###
###############################

time_unit = np.sqrt(M * sigma**2/epsilon)

velocity_unit = sigma / time_unit
dipole_unit = M**(1/2) * sigma**(5/2) * time_unit**-1 #statC . cm

########################################
### Calculating reduced LJ parameter ###
########################################

sigma_Ar_Ar = 3.410 / sigma_
sigma_Ar_Xe = 3.735 / sigma_
sigma_Xe_Xe = 4.060 / sigma_

epsilon_Ar_Ar = 0.996 / epsilon_
epsilon_Ar_Xe = 1.377 / epsilon_
epsilon_Xe_Xe = 1.904 / epsilon_

mass_dict = {"Ar": 39.948 / M_ , "Xe" : 131.293 / M_}

############################################
### Calculating reduced dipole parameter ###
############################################

bohr_rad, _, _ = physical_constants["Bohr radius"]

mu0 = (0.0124 * e_charge * bohr_rad * 10 * 3e10) / ( dipole_unit)

bohr_rad *= 1e2

a = 1.5121 * sigma / bohr_rad

d0 = 7.10 * bohr_rad / sigma

#####################################
### Calculating reduced constants ###
#####################################

c = 3e10 / (sigma / time_unit)

#####################
### FUNCTIONALITY ###
#####################

def generate_LJparam_matrix(idxAr, idxXe):
    epsilon_mat = (np.outer(idxAr,idxAr) * epsilon_Ar_Ar \
        + np.outer(idxAr, idxXe) * epsilon_Ar_Xe \
        + np.outer(idxXe, idxAr) * epsilon_Ar_Xe \
        + np.outer(idxXe, idxXe) * epsilon_Xe_Xe )

    sigma_mat = (np.outer(idxAr,idxAr) * sigma_Ar_Ar \
        + np.outer(idxAr, idxXe) * sigma_Ar_Xe \
        + np.outer(idxXe, idxAr) * sigma_Ar_Xe \
        + np.outer(idxXe, idxXe) * sigma_Xe_Xe) 

    return epsilon_mat, sigma_mat

if test:
    print("Epsilon (erg)", epsilon)
    print("Reduced epsilon (Ar-Ar, Ar-Xe, Xe-Xe):", 
            epsilon_Ar_Ar,";", epsilon_Ar_Xe,";", epsilon_Xe_Xe)

    print("######################")
    print("Length multiple - Sigma (cm)", sigma)
    print("Reduced sigma (Ar-Ar, Ar-Xe, Xe-Xe):", 
            sigma_Ar_Ar,";", sigma_Ar_Xe,";", sigma_Xe_Xe)

    print("######################")
    print("Mass multiples (g)",M)
    print("Reduced Mass (Ar, Xe):", mass_dict["Ar"],";",mass_dict["Xe"])

    print("######################")
    print("Dipole unit multiple (statC . cm)", dipole_unit)
    print("Reduced dipole parameter:")
    print("mu0 = ",mu0)
    print("a = ",a)
    print("d0 = ",d0)

    print("######################")
    print("Time", time_unit)
    print("Velocity multiple (cm/s)", velocity_unit)

    print("######################")
    print("c: ", c)

