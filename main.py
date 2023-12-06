import numpy as np
import matplotlib.pyplot as plt

class BaseMolecularDynamic:
    def __init__(self):
        self.q = []
        self.E = []

    def compute_force(self):
        pass

    def integrate(self, force, dt):
        pass

    def compute_energy(self, new_q, v):
        pass

    def simulate(self, dt, steps):
        for i in range(steps):
            force = self.compute_force()
            q, v = self.integrate(
                    force = force, dt =  dt)
            self.q.append(q)
            E = self.compute_energy(q = q, v = v)
            self.E.append(E)


class PrototypeMD(BaseMolecularDynamic):
    def __init__(self, N, L = 10, q0 = None):

        super().__init__()

        if isinstance(q0, np.ndarray):
            q0 = q0
        else:
            assert q0 == None
            q0 = np.random.uniform(
                    low = 0, high = L, 
                    size = N)

        self.q.append(q0)

        self.v0 = np.random.uniform(
                low = -0.5, high = 0.5,
                size = N)

    def compute_force(self):
        q = self.q[-1]
        return 2 * q - 2 * q**3

    def integrate(self,force,dt):
        q = self.q[-1]

        if len(self.q) < 2:
            new_q = q + self.v0 * dt + force * dt**2
            v = (new_q - q) / dt

        else:
            q_ = self.q[-2]
            new_q = 2*q - q_ + force * dt**2
            v = (new_q - q_) / (2*dt)

        return new_q , v


    def compute_energy(self, q, v):
        return v**2 + 0.5 - q**2  + 0.5 * q**4


md = PrototypeMD(N = 2, L = 2, q0 = np.array([2,5]))
N_steps = 300; dt = 0.05
md.simulate(dt = dt, steps = N_steps)

fig, ax = plt.subplots()

ax.plot(np.arange(N_steps + dt) * dt, md.q)

fig.savefig("trajectory.jpeg",dpi = 600)

fig, ax = plt.subplots()

ax.plot(np.arange(dt, N_steps + dt) * dt, md.E)

fig.savefig("energy.jpeg",dpi = 600)
