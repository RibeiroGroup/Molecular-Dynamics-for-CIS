import numpy as np

class ExplicitRungeKutta4:
    def __init__(self, func):
        self.func = func

    def update_func(self,func):
        self.func = func

    def step(self, X, h):
        k1 = self.func(X)
        k2 = self.func(X + k1 * h/2)
        k3 = self.func(X + k2 * h/2)
        k4 = self.func(X + k3 * h)

        X = X + (k1 + 2*k2 + 2*k3 + k4)*h/6

        return X

        
