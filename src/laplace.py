import numpy as np

import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as kb, Planck as h

def emission_laplace(beta, v):
    return v ** 4 * np.exp(-beta * v)

fig,ax = plt.subplots()

x = np.linspace(0,1500, 1500)
ax.plot(x, emission_laplace(0.01,x))

fig.savefig("spectral.jpeg",dpi = 600)
