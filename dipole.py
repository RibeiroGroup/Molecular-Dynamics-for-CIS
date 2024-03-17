import numpy as np
import matplotlib.pyplot as plt
import sympy as sm

class DipoleFunction:
    def __init__(self, parameters, engine = "Grigoriev"):
        if engine == "Grigoriev":
            assert "mu0" in parameters.keys()
            assert "a" in parameters.keys()
            assert "R0" in parameters.keys()
            assert "D7" in parameters.keys()

        self.engine = engine
        self.parameters = parameters

    def grigoriev_dipole_function(self, distance):
        mu0 = self.parameters["mu0"]
        a = self.parameters["a"]
        R0 = self.parameters["R0"]
        D7 = self.parameters["D7"]
        return mu0 * np.exp(-a*(distance-R0)) - D7/(R**7)

    def __call__(self, distance, distance_vec):
        if self.engine == "grigoriev":
            return self.grigoriev_dipole_function(distance)
        else:
            raise Exception("Dipole function not found")

