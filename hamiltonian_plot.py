import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("result_plot/sim_result.pkl","rb") as handle:
    sim_result = pickle.load(handle)

print(sim_result.keys())

with open("result_plot/sim_result2.pkl","rb") as handle:
    sim_result2 = pickle.load(handle)

fig, ax = plt.subplots(2,3,figsize= (18,6))

plot_range = slice(0, 60000)# len(sim_result["steps"]))

steps_list = np.array(sim_result["steps"][plot_range]) * sim_result["h"]

"""
PLOTTING EM FIELD HAMILTONIAN
"""
H_em = np.array(sim_result["em"])
print(H_em.shape[1])
for i in range(H_em.shape[1]):
    ax[0,0].plot(steps_list, H_em[plot_range,i],
            label = ["x", "y", "z", "xy"][i] )

ax[0,0].set_ylabel("H, field")
ax[0,0].legend()

ax[1,0].plot(steps_list, sim_result["mat"][plot_range],c="c")
#ax[1,0].plot(steps_list, sim_result2["mat"][plot_range],c="r",linestyle="--")
ax[1,0].set_ylabel("H, matter")
ax[1,0].set_xlabel("Absolute time")

"""
PLOTTING OSCILLATOR HAMILTONIAN
"""
ax[1,1].plot(steps_list, sim_result["osci"][plot_range],c="c")
#ax[1,1].plot(steps_list, sim_result2["osci"][plot_range],c="r",linestyle="--")
ax[1,1].set_ylabel("H, oscillator")
ax[1,1].set_xlabel("Absolute time")

ax[1,2].plot(steps_list, sim_result["amplitude"][plot_range],c="c")
#ax[1,2].plot(steps_list, sim_result2["amplitude"][plot_range],c="r",linestyle="--")
ax[1,2].set_ylabel(r"Amplitude $|x|$")
ax[1,2].set_xlabel("Absolute time")

H_list = np.array(sim_result["mat"][plot_range]) \
        + np.sum(np.array(sim_result["em"][plot_range]),axis=1) \
        + np.array(sim_result["osci"][plot_range])

ax[0,1].plot(steps_list, H_list,c="c")
ax[0,2].plot([],[],c="c")
ax[0,2].plot([],[],c="r")
ax[0,1].set_ylabel("H, total for HO in EM")

"""
H_list2 = np.array(sim_result2["mat"][plot_range]) \
        + np.array(sim_result2["em"][plot_range]) \
        + np.array(sim_result2["osci"][plot_range])

ax[0,2].plot(steps_list, H_list2,c="r",linestyle="--")
"""
ax[0,2].plot([],[],c="c",label = "HO in EM field")
ax[0,2].plot([],[],c="r",label = "Ordinary HO")
ax[0,2].legend()
ax[0,2].set_ylabel("H, total for ordinary HO")

fig.savefig("result_plot/particle_field_energy.jpeg",dpi=600,bbox_inches="tight")

"""
PLOTTING OSCILLATOR VELOCITY

fig, ax = plt.subplots()

velocity = np.array(sim_result["v"])[plot_range,0,0]
ax.plot(steps_list, velocity, label = r"$v_k$")
velocity = np.array(sim_result["v"])[plot_range,0,1]
ax.plot(steps_list, velocity, label = r"$v_{k1}$")
velocity = np.array(sim_result["v"])[plot_range,0,2]
ax.plot(steps_list, velocity, label = r"$v_{k2}$", linestyle = "dashed")

ax.legend()
fig.savefig("result_plot/velocity.jpeg",dpi=500)
"""

"""
PLOTTING OSCILLATOR POSITION

fig, ax = plt.subplots()

velocity = np.array(sim_result["r"])[plot_range,0,0]
ax.plot(steps_list, velocity, label = r"$r_k$")
velocity = np.array(sim_result["r"])[plot_range,0,1]
ax.plot(steps_list, velocity, label = r"$r_{k1}$")
velocity = np.array(sim_result["r"])[plot_range,0,2]
ax.plot(steps_list, velocity, label = r"$r_{k2}$", linestyle = "dashed")

ax.legend()
fig.savefig("result_plot/position.jpeg",dpi=500)

fig, ax = plt.subplots(2)
C = np.array(sim_result["C"])
C = C[plot_range, :]
ax[0].plot(steps_list, np.real(C[:,0]), color = "blue", label = r"real($C_{k1}$)", linestyle = "dotted")
ax[0].plot(steps_list, np.imag(C[:,0]), color = "blue", label = r"imag($C_{k1}$)")

ax[1].plot(steps_list, np.real(C[:,1]), color = "red", label = r"real($C_{k2}$)", linestyle = "dotted")
ax[1].plot(steps_list, np.imag(C[:,1]), color = "red", label = r"imag($C_{k2}$)")

ax[0].legend()
ax[1].legend()
fig.savefig("result_plot/EM_coef.jpeg",dpi=500)
"""
