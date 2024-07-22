import glob
import tqdm
import os, sys, pickle, copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import utilities.reduced_parameter as red

from analyze_tools.monte_carlo import get_colliding_time

#PICKLE_PATH = "/home/ribeirogroup/code/mm_polariton_pickle_ja/result_Jul12nd_2024_0939/*"
#PICKLE_PATH = "pickle_jar/result_Jul13th_2024_0956/*"
#PICKLE_PATH = "pickle_jar/result_Jul15th_2024_0956/*"
PICKLE_PATH = "pickle_jar/*"

fig1,ax1 = plt.subplots(2,figsize = (6,8))
#fig2,ax2 = plt.subplots(2,figsize = (6,8))

collision_time = {"cavity":[], "free":[]}
initial_velocity_data = {
        "cavity":{"Ar": [], "Xe":[]},
        "free":{"Ar": [], "Xe":[]}
        }
traj_loc = {"cavity":[], "free":[]}

for j,KEYWORDS in enumerate(["cavity","free"]):

    ar_angle = []
    xe_angle = []

    magnitude = lambda vec: np.sqrt(np.einsum("ni,ni->n",vec,vec))

    for PATH in glob.glob(PICKLE_PATH):
        
        if KEYWORDS not in PATH: continue

        with open(PATH,"rb") as handle:
            result = pickle.load(handle)

        atoms = copy.deepcopy(result["atoms"])
        mu0 = result["mu0"]

        traj_len = len(atoms.trajectory["r"])

        initial_velocity_data[KEYWORDS]["Ar"] += \
                list(magnitude(atoms.trajectory["r_dot"][0][:N_pairs]))
        initial_velocity_data[KEYWORDS]["Xe"] += \
                list(magnitude(atoms.trajectory["r_dot"][0][N_pairs:]))

        t = get_colliding_time(atoms, mu0 = mu0, dipole_threshold = 1e-3)

        collision_time[KEYWORDS] = t

        traj_loc[KEYWORDS].append((PATH, i))

initial_arvelocity = initial_velocity_data["cavity"]["Ar"]
initial_xevelocity = initial_velocity_data["cavity"]["Xe"]

delta_collision_time = np.array(collision_time["cavity"])\
    - np.array(collision_time["free"])

ax1[0].scatter(initial_arvelocity,delta_collision_time)
ax1[1].scatter(initial_xevelocity,delta_collision_time)

fig1.savefig("figure2/delta_collision_time.jpeg",dpi=600)

time_sorted_cavtraj = np.argsort(collision_time["cavity"])

for j in [1,2,3]:
    PATH, i = traj_loc["cavity"][time_sorted_cavtraj[-j]]
    print(PATH)

    with open(PATH,"rb") as handle:
        result = pickle.load(handle)

    PATH = PATH.replace('cavity','free')
    print(PATH)

    with open(PATH,"rb") as handle:
        result0 = pickle.load(handle)

    atoms = result["atoms"]

    atoms0 = result0["atoms"]
    traj_len0 = len(atoms0.trajectory["r"])

    fig, ax = plt.subplots(2,2,figsize = (12,12))

    x_ar = atoms.trajectory["r"][0][i][0]
    y_ar = atoms.trajectory["r"][0][i][1]
    z_ar = atoms.trajectory["r"][0][i][2]
    ar_posxy, = ax[0][0].plot([x_ar], [y_ar], 'ro', markersize = 5)
    ar_posyz, = ax[0][1].plot([y_ar], [z_ar], 'ro', markersize = 5)

    x_xe = atoms.trajectory["r"][0][i+N_pairs][0]
    y_xe = atoms.trajectory["r"][0][i+N_pairs][1]
    z_xe = atoms.trajectory["r"][0][i+N_pairs][2]
    xe_posxy, = ax[0][0].plot([x_xe], [y_xe], 'ro', markersize = 10)
    xe_posyz, = ax[0][1].plot([y_xe], [z_xe], 'ro', markersize = 10)

    ax[0][0].set_xlim(np.min([x_ar,x_xe]) - 5, np.max([x_ar,x_xe]) + 5)
    ax[0][0].set_ylim(np.min([y_ar,y_xe]) - 5, np.max([y_ar,y_xe]) + 5)
    ax[0][1].set_xlim(np.min([y_ar,y_xe]) - 5, np.max([y_ar,y_xe]) + 5)
    ax[0][1].set_ylim(np.min([z_ar,z_xe]) - 5, np.max([z_ar,z_xe]) + 5)

    x_ar = atoms0.trajectory["r"][0][i][0]
    y_ar = atoms0.trajectory["r"][0][i][1]
    z_ar = atoms0.trajectory["r"][0][i][2]
    ar_posxy0, = ax[1][0].plot([x_ar], [y_ar], 'ro', markersize = 5)
    ar_posyz0, = ax[1][1].plot([y_ar], [z_ar], 'ro', markersize = 5)

    x_xe = atoms0.trajectory["r"][0][i+N_pairs][0]
    y_xe = atoms0.trajectory["r"][0][i+N_pairs][1]
    z_xe = atoms0.trajectory["r"][0][i+N_pairs][2]
    xe_posxy0, = ax[1][0].plot([x_xe], [y_xe], 'ro', markersize = 10)
    xe_posyz0, = ax[1][1].plot([y_xe], [z_xe], 'ro', markersize = 10)

    ax[1][0].set_xlim(np.min([x_ar,x_xe]) - 5, np.max([x_ar,x_xe]) + 5)
    ax[1][0].set_ylim(np.min([y_ar,y_xe]) - 5, np.max([y_ar,y_xe]) + 5)
    ax[1][1].set_xlim(np.min([y_ar,y_xe]) - 5, np.max([y_ar,y_xe]) + 5)
    ax[1][1].set_ylim(np.min([z_ar,z_xe]) - 5, np.max([z_ar,z_xe]) + 5)

    def update(frame):
        frame = frame * 5
        x_ar = atoms.trajectory["r"][frame][i][0]
        y_ar = atoms.trajectory["r"][frame][i][1]
        z_ar = atoms.trajectory["r"][frame][i][2]
        ar_posxy.set_xdata([x_ar])
        ar_posxy.set_ydata([y_ar])

        ar_posyz.set_xdata([y_ar])
        ar_posyz.set_ydata([z_ar])

        x_xe = atoms.trajectory["r"][frame][i+N_pairs][0]
        y_xe = atoms.trajectory["r"][frame][i+N_pairs][1]
        z_xe = atoms.trajectory["r"][frame][i+N_pairs][2]
        xe_posxy.set_xdata([x_xe])
        xe_posxy.set_ydata([y_xe])

        xe_posyz.set_xdata([y_xe])
        xe_posyz.set_ydata([z_xe])

        ax[0][0].set_xlim(np.min([x_ar,x_xe]) - 5, np.max([x_ar,x_xe]) + 5)
        ax[0][0].set_ylim(np.min([y_ar,y_xe]) - 5, np.max([y_ar,y_xe]) + 5)
        ax[0][1].set_xlim(np.min([y_ar,y_xe]) - 5, np.max([y_ar,y_xe]) + 5)
        ax[0][1].set_ylim(np.min([z_ar,z_xe]) - 5, np.max([z_ar,z_xe]) + 5)

        frame = np.min([frame, traj_len0-1])

        x_ar = atoms0.trajectory["r"][frame][i][0]
        y_ar = atoms0.trajectory["r"][frame][i][1]
        z_ar = atoms0.trajectory["r"][frame][i][2]
        ar_posxy0.set_xdata([x_ar])
        ar_posxy0.set_ydata([y_ar])

        ar_posyz0.set_xdata([y_ar])
        ar_posyz0.set_ydata([z_ar])

        x_xe = atoms0.trajectory["r"][frame][i+N_pairs][0]
        y_xe = atoms0.trajectory["r"][frame][i+N_pairs][1]
        z_xe = atoms0.trajectory["r"][frame][i+N_pairs][2]
        xe_posxy0.set_xdata([x_xe])
        xe_posxy0.set_ydata([y_xe])

        xe_posyz0.set_xdata([y_xe])
        xe_posyz0.set_ydata([z_xe])

        ax[1][0].set_xlim(np.min([x_ar,x_xe]) - 5, np.max([x_ar,x_xe]) + 5)
        ax[1][0].set_ylim(np.min([y_ar,y_xe]) - 5, np.max([y_ar,y_xe]) + 5)
        ax[1][1].set_xlim(np.min([y_ar,y_xe]) - 5, np.max([y_ar,y_xe]) + 5)
        ax[1][1].set_ylim(np.min([z_ar,z_xe]) - 5, np.max([z_ar,z_xe]) + 5)

        return (
                ar_posxy, ar_posyz, xe_posxy, xe_posyz,
                ar_posxy0, ar_posyz0, xe_posxy0, xe_posyz0
                )

    ani = animation.FuncAnimation(
            fig=fig, func=update, frames=int(np.floor(traj_len/5)), interval=10)
    ani.save(filename="figure2/trajectory_"+str(j)+".html", writer="html")
