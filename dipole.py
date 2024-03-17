import numpy as np

from distance import DistanceCalculator

class BaseDipoleFunction:
    def __init__(self, distance_calc):

        self.distance_calc = distance_calc

    def __call__(self, distance, distance_vec):

        return self.distance_calc.apply_function(dipole_func)

    def gradient(self,distance, distance_vec):

        return self.distance_calc.apply_function(gradient_func)


class SimpleDipoleFunction(BaseDipoleFunction):
    def __init__(self, distance_calc, mu0, a, d0):

        super().__init__()
        self.mu0 = mu0
        self.a = a
        self.d0 = d0

    def dipole_func(self, distance, distance_vec):
        
