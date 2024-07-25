import warnings
warnings.filterwarnings('ignore')

import argparse
import os, sys
from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utilities import reduced_parameter as red
from field.utils import profiling_rad

import utilities.reduced_parameter as red

from utilities.etc import categorizing_result
from analyze_tools.plot_tools import Plot

#ROOT = "pickle_jar/"
ROOT = "/Users/macbook/OneDrive - Emory/Research data/mm_polariton/pickle_jar/"

temperature_list = []

temperature = 292 ; seed = 157 ; N_atoms = 512
temperature_list.append(temperature)

free292 = "{}_{}_25_40_{}".format(temperature, N_atoms, seed)
cav_result_folders_list292 = [
    "{}_{}_25_40_{}".format(temperature, N_atoms, seed),
    #"{}_{}_40_60_{}".format(temperature, N_atoms, seed),
    #"{}_{}_60_80_{}".format(temperature, N_atoms, seed),
]

temperature = 200 ; seed = 715 ; N_atoms = 512
temperature_list.append(temperature)

free200 = "{}_{}_25_40_{}".format(temperature, N_atoms, seed)
cav_result_folders_list200 = [
    "{}_{}_25_40_{}".format(temperature, N_atoms, seed),
    #"{}_{}_40_60_{}".format(temperature, N_atoms, seed),
    #"{}_{}_60_80_{}".format(temperature, N_atoms, seed),
]

"""
for folder in cav_result_folders_list:
    result_dict = categorizing_result(ROOT + folder)  
    cav_result_dict_list.append(result_dict)
"""

free_plot = Plot(n_spec_plots = 3)

for i , free_result_folders in enumerate([free292, free200]):
    result_dict = categorizing_result(ROOT + free_result_folders, KEYWORDS="free")

    cycle = 19
    Afield = result_dict[cycle]["probe_field"]

    rad_energy = red.convert_energy(np.array(Afield.history["energy"][-1]), "ev")
    omega = red.convert_wavenumber(Afield.k_val)
    omega_profile, final_rad_profile = profiling_rad(omega, rad_energy)

    free_plot.add_spec_plot(
        0, omega_profile, final_rad_profile, ma_w = 20, 
        line_label = "T = {}K".format(temperature_list[i])
        )

    for cycle in range(4,20,5):
        Afield = result_dict[cycle]["probe_field"]

        rad_energy = red.convert_energy(np.array(Afield.history["energy"][-1]), "ev")
        omega = red.convert_wavenumber(Afield.k_val)
        omega_profile, final_rad_profile = profiling_rad(omega, rad_energy)

        free_plot.add_spec_plot(
            i + 1, omega_profile, final_rad_profile, ma_w = 20, scatter = False,
            line_label = "after {} cycles".format(cycle + 1))

free_plot.add_label(0, (None, "Radiation energy (eV)"))
free_plot.add_legend(0)
free_plot.add_label(1, (None, "Radiation energy (eV)"))
free_plot.add_legend(1)
free_plot.add_label(2)

free_plot.annotate()
free_plot.add_legend(2)

free_plot.savefig("figure/test/free2_field_")
