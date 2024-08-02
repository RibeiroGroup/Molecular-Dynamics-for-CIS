from copy import deepcopy
import numpy as np

import scipy.constants as scicon
from scipy.constants import physical_constants, Avogadro, e as e_charge, hbar as hbar_original

test = 0

c = 1.37e2
boltzmann = 1.380649e-16 #UNIT: (erg/K)

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

#mu0 = (0.0124 * e_charge * bohr_rad * 3e11) / ( dipole_unit)
mu0 = (0.0284 * e_charge * bohr_rad * 3e11) / ( dipole_unit)

bohr_rad *= 1e2

#a = 1.5121 * sigma / bohr_rad
a = 1.22522 * sigma / bohr_rad

d0 = 7.10 * bohr_rad / sigma
d7 = 14200 * (bohr_rad / sigma) ** 7

#####################################
### Calculating reduced constants ###
#####################################

c = 3e10 / (sigma / time_unit)

temp = epsilon / boltzmann

hbar = hbar_original * 1e2 ** 2 * 1e3 #convert to cgs
hbar = hbar * time_unit / (M * sigma**2)

#####################
### FUNCTIONALITY ###
#####################

def generate_LJparam_matrix(idxAr, idxXe):
    """
    Generate matrix of Lennard-Jones parameters for evaluating 
    force
    """
    epsilon_mat = (np.outer(idxAr,idxAr) * epsilon_Ar_Ar \
        + np.outer(idxAr, idxXe) * epsilon_Ar_Xe \
        + np.outer(idxXe, idxAr) * epsilon_Ar_Xe \
        + np.outer(idxXe, idxXe) * epsilon_Xe_Xe )

    sigma_mat = (np.outer(idxAr,idxAr) * sigma_Ar_Ar \
        + np.outer(idxAr, idxXe) * sigma_Ar_Xe \
        + np.outer(idxXe, idxAr) * sigma_Ar_Xe \
        + np.outer(idxXe, idxXe) * sigma_Xe_Xe) 

    return epsilon_mat, sigma_mat

def convert_energy(array, unit):
    """
    Convert energy array from r.u. to unit of choice
    Args: 
    + array (np.array): array of energy
    + unit (str): either 'ev' or 'cm-1'
    """
    array = deepcopy(array) * (epsilon * 6.242e11)
    if unit == "ev":
        return array
    elif unit == "cm-1":
        array *= 8065.56
        return array
    else:
        raise Exception(
            'The specified unit is not supported. Please specify either "ev" or "cm-1"!')

def convert_wavenumber(array):
    # convert wavenumber array from r.u. to 1/cm
    array = deepcopy(array) / (sigma * 2 * np.pi)
    return array

def convert_time(array):
    # convert time array in r.u. to ps
    array = deepcopy(np.array(array)) * time_unit * 1e12
    return array

def convert_dipole(array):
    """
    Convert Dipole from reduced unit to Debye
    """
    print("FYI, this function divide the dipole by 2 to make up for duoble counting of dipole")
    array = deepcopy(np.array(array)) * dipole_unit * 1e18
    return array

if test:
    import os, sys
    parent_path = os.path.abspath("")
    sys.path.append(parent_path)

    #from src.dipole import Grigoriev_dipole
    #from src.forcefield import LJ_potential

    print("Epsilon (erg)", epsilon)
    print("Reduced epsilon (Ar-Ar, Ar-Xe, Xe-Xe):", 
            epsilon_Ar_Ar,";", epsilon_Ar_Xe,";", epsilon_Xe_Xe)
    print("CGS epsilon (Ar-Ar, Ar-Xe, Xe-Xe):", 
            epsilon_Ar_Ar * epsilon,";", epsilon_Ar_Xe * epsilon,";", epsilon_Xe_Xe * epsilon)

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

    print("CGS dipole parameter:")
    print("mu0 = ",mu0 * dipole_unit)
    print("a = {:.2E}".format(a / sigma))
    print("d0 = ",d0 * sigma)

    print("######################")
    print("Time", time_unit)
    print("Velocity multiple (cm/s)", velocity_unit)

    print("######################")
    print("c: ", c)

    emat, smat = generate_LJparam_matrix([1,0],[0,1])
    print("Epsilon parameters matrix ")
    print(emat)
    print("Sigma parameters matrix ")
    print(smat)

    print(hbar)

    """
    fig,ax = plt.subplots(1,2,figsize = (12,4))

    dist_list = np.linspace(1,3,100)

    def LJ_potential(sigma, epsilon, distance):

        V = 4 * epsilon * ( (sigma/distance)**12 - (sigma/distance)**6 )

        return V

    potential = list(
        map(lambda d: LJ_potential(sigma_Ar_Ar, epsilon_Ar_Ar, d), dist_list))
    ax[0].plot(dist_list, potential,label = "Ar-Ar potential")

    potential = list(
        map(lambda d: LJ_potential(sigma_Ar_Xe, epsilon_Ar_Xe, d), dist_list))
    ax[0].plot(dist_list, potential,label = "Ar-Xe potential")
    
    potential = list(
        map(lambda d: LJ_potential(sigma_Xe_Xe, epsilon_Xe_Xe, d), dist_list))
    ax[0].plot(dist_list, potential,label = "Xe-Xe potential")

    dist_list = np.linspace(1,3,100)

    dipole = list(
        map(lambda d: mu0 * np.exp(-a * (d - d0)), dist_list
        )) 
    ax[1].plot(dist_list, dipole,label = "Ar-Xe Dipole")

    ax[0].legend()
    fig.savefig("figure/parameter_visual.jpeg",dpi = 600)

    """
