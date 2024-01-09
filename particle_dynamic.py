import numpy as np

box_dim = np.ones(3) * 3 

r = np.zeros(3)
v = np.random.rand(3)

trajectory = {"r":[r],"v":[v]}

h = 1e-3

for i in range(int(1e3 + 1)):
    k1 = v
    k2 = v + k1 * h/2
    k3 = v + k2 * h/3
    k4 = v + k4 * h

    dr = (h*6) * (k1 + 2*k2 + 2*k3 + k4)
    r += dr
    trajectory.append(r)
    trajectory.append(v)


