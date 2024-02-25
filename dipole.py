import numpy as np
import matplotlib.pyplot as plt
import sympy as sm

class DipoleFunction:
    def __init__(self, parameters, engine = "morse"):
        if engine == "morse":
            assert "mu0" in parameters.keys()
            assert "Re" in parameters.keys()
            assert "L" in parameters.keys()

        elif engine == "Grigoriev":
            assert "mu0" in parameters.keys()
            assert "a" in parameters.keys()
            assert "R0" in parameters.keys()
            assert "D7" in parameters.keys()

        self.engine = engine
        self.parameters = parameters

    def morse_dipole_function(self, R):
        mu0 = self.parameters["mu0"]
        Re = self.parameters["Re"]
        L = self.parameters["L"]
        return mu0 * np.exp(2 * Re / L) * np.exp(-2 * R / L)

    def grigoriev_dipole_function(self,R):
        mu0 = self.parameters["mu0"]
        a = self.parameters["a"]
        R0 = self.parameters["R0"]
        D7 = self.parameters["D7"]
        return mu0 * np.exp(-a*(R-R0)) - D7/(R**7)

    def markus_dipole_function(self,R):
        mu0 = self.parameters["mu0"]
        a = self.parameters["a"]
        R0 = self.parameters["R0"]
        D7 = self.parameters["D7"]

    def __call__(self,R):
        if self.engine == "morse":
            return self.morse_dipole_function(R)
        elif self.engine == "grigoriev":
            return self.grigoriev_dipole_function(R)
        else:
            raise Exception

"""
morse_dipole_func = DipoleFunction(
        parameters = {
            "mu0":1e5, "Re":1.377 * 1.59360e-3, 
            "L":(1e-10 / 5.29177e-11) / 3.735
            }
    )

gri_dipole_func = DipoleFunction(
        parameters = {
            "mu0":1, "R0": 7.10, "a":1.5121, "D7":300},
        engine = "grigoriev"
)

X = np.arange(2,10,0.1)

fig, ax = plt.subplots()

#ax.plot(X, morse_dipole_func(X),label = "morse")
ax.plot(X, gri_dipole_func(X),label = "grigoriev")

fig.legend()

fig.savefig("result_plot\others\dipole_plot.jpeg",dpi = 300)

"""

