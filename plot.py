import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("result_plot/trajectory.pkl","rb") as handle:
    trajectory = pickle.load(handle)

plot_range = slice(0, len(trajectory["steps"]))

t = np.array(trajectory["steps"][plot_range])* trajectory["h"]

fig, ax = plt.subplots(3)
ax[0].plot(t, trajectory["H"][plot_range])
ax[1].plot(t, trajectory["T"][plot_range])
ax[2].plot(t, trajectory["V"][plot_range])

fig.savefig("result_plot/hamiltonians.jpeg", bbox_inches="tight",dpi=600)
