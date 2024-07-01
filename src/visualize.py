
fig, ax = plt.subplots(2,2,figsize = (12,8))

ax[0,0].plot(tlist, total_energy)
ax[0,0].set_ylabel("Total energy")

ax[0,1].plot(tlist, kinetic_elist)
ax[0,1].set_ylabel("Kinetic energy")

ax[1,0].plot(tlist, potential_elist)
ax[1,0].set_ylabel("Potential energy")

ax[1,1].plot(tlist, field_elist)
ax[1,1].set_ylabel("Radiation energy")

fig.savefig("figure/full_simulation.jpeg",dpi = 600)
