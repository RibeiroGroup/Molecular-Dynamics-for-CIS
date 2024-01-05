import numpy as np

class RK4:
    def __init__(self, func, h, x0, y0):
        self.func = func
        self.h = h
        self.x = [x0]
        self.y = [y0]

    def step(self):
        x = self.x[-1]
        y = self.y[-1]
        h = self.h

        k1 = self.func(x,y)
        k2 = self.func(x + h/2, y + h*k1/2)
        k3 = self.func(x + h/2, y + h*k2/2)
        k4 = self.func(x + h  , y + h*k3)

        x_new = x + h
        y_new = y + self.h * (k1 + 2*k2 + 2*k3 + k4)/6
        self.x.append(x_new)
        self.y.append(y_new)

        return x_new, y_new


