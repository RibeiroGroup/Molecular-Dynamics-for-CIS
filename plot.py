import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("result_plot/trajectory.pkl","rb") as handle:
    trajectory = pickle.load(handle)

plot_range = slice(0, len(trajectory["steps"]))

fig, ax = plt.subplots(3)
ax[0].plot(trajectory["steps"][plot_range], trajectory["H"][plot_range])
ax[1].plot(trajectory["steps"][plot_range], trajectory["T"][plot_range])
ax[2].plot(trajectory["steps"][plot_range], trajectory["V"][plot_range])
fig.savefig("foo.jpeg",dpi=600)
