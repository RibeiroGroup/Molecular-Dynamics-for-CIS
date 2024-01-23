import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("result_plot/sim_result.pkl","rb") as handle:
    sim_result = pickle.load(handle)

with open("result_plot/sim_result2.pkl","rb") as handle:
    sim_result2 = pickle.load(handle)

fig, ax = plt.subplots(2,3,figsize= (18,6))

plot_range = slice(0, 10000)# len(sim_result["steps"]))

steps_list = np.array(sim_result["steps"][plot_range]) * sim_result["h"]

ax[0,0].plot(steps_list, sim_result["em"][plot_range],c="c")
ax[0,0].plot(steps_list, sim_result2["em"][plot_range],c="r",linestyle="--")
ax[0,0].set_ylabel("H, field")

ax[1,0].plot(steps_list, sim_result2["mat"][plot_range],c="r",linestyle="--")
ax[1,0].plot(steps_list, sim_result["mat"][plot_range],c="c")
ax[1,0].set_ylabel("H, matter")
ax[1,0].set_xlabel("Absolute time")

ax[1,1].plot(steps_list, sim_result2["osci"][plot_range],c="r",linestyle="--")
ax[1,1].plot(steps_list, sim_result["osci"][plot_range],c="c")
ax[1,1].set_ylabel("H, oscillator")
ax[1,1].set_xlabel("Absolute time")

ax[1,2].plot(steps_list, sim_result2["amplitude"][plot_range],c="r",linestyle="--")
ax[1,2].plot(steps_list, sim_result["amplitude"][plot_range],c="c")
ax[1,2].set_ylabel(r"Amplitude $|x|$")
ax[1,2].set_xlabel("Absolute time")

H_list = np.array(sim_result["mat"][plot_range]) \
        + np.array(sim_result["em"][plot_range]) \
        + np.array(sim_result["osci"][plot_range])

ax[0,1].plot(steps_list, H_list,c="c")
ax[0,2].plot([],[],c="c")
ax[0,2].plot([],[],c="r")
ax[0,1].set_ylabel("H, total for HO in EM")

H_list2 = np.array(sim_result2["mat"][plot_range]) \
        + np.array(sim_result2["em"][plot_range]) \
        + np.array(sim_result2["osci"][plot_range])

ax[0,2].plot(steps_list, H_list2,c="r",linestyle="--")
ax[0,2].plot([],[],c="c",label = "HO in EM field")
ax[0,2].plot([],[],c="r",label = "Ordinary HO")
ax[0,2].legend()
ax[0,2].set_ylabel("H, total for ordinary HO")

fig.savefig("result_plot/particle_field_energy.jpeg",dpi=600,bbox_inches="tight")

