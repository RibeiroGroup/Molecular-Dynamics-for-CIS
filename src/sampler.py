import numpy as np
import matplotlib.pyplot as plt

from matter.atoms import AtomsInBox
from matter.utils import AllInOneSampler

import utilities.reduced_parameter as red

N_atom_pairs = 10000
L = 1e6
K_temp = 298

sampler = AllInOneSampler(
        N_atom_pairs=N_atom_pairs, angle_range=np.pi/4, L=L,
        d_ar_xe=4,red_temp_unit=red.temp, K_temp=K_temp,
        ar_mass=red.mass_dict["Ar"], xe_mass=red.mass_dict["Xe"]
        )

sample = sampler()
r_ar, r_xe = sample["r"]
r_dot_ar, r_dot_xe = sample["r_dot"]

r_dot_ar = np.sqrt(np.einsum("ni,ni->n",r_dot_ar,r_dot_ar)) \
     * red.velocity_unit * 1e-2 * 1e-3
r_dot_xe = np.sqrt(np.einsum("ni,ni->n",r_dot_xe,r_dot_xe)) \
     * red.velocity_unit * 1e-2 * 1e-3

fig, ax = plt.subplots(1,2,figsize = (12,4))

n,_,_ = ax[1].hist(r_dot_xe, bins = np.arange(0,1.25,0.01))
ax[0].hist(r_dot_ar, bins = np.arange(0,1.25,0.01))
ax[0].set_ylim(0, np.max(n))

fig.savefig("figure/velocity_profile/temp_{}.jpeg".format(K_temp),dpi = 600,bbox_inches="tight")
