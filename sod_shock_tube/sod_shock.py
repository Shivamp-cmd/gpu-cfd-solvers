import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0          # domain length
Nx = 500         # number of spatial points
dx = L / Nx
c = 1.0          # wave speed
CFL = 0.9        # CFL number (must be <1)
dt = CFL * dx / c
Nt = 600         # number of time steps

# Spatial grid
x = cp.linspace(0, L, Nx)

# Allocate wave arrays
u = cp.zeros(Nx)       # current time step
u_prev = cp.zeros(Nx)  # previous
u_next = cp.zeros(Nx)  # next

# Initial condition: Gaussian pulse
u[:] = cp.exp(-1000 * (x - 0.5)**2)
u_prev[:] = u[:]  # assume initial velocity is 0

# Time stepping
for n in range(Nt):
    u_next[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + \
        (CFL**2) * (u[2:] - 2*u[1:-1] + u[:-2])

    # Reflecting BCs
    u_next[0] = 0
    u_next[-1] = 0

    # Shift time steps
    u_prev[:] = u[:]
    u[:] = u_next[:]

    # Optional: plot every 100 steps
    if n % 100 == 0:
        plt.clf()
        plt.plot(cp.asnumpy(x), cp.asnumpy(u), label=f"t = {n*dt:.3f}")
        plt.ylim(-1.2, 1.2)
        plt.title("1D Wave Equation (GPU - CuPy)")
        plt.xlabel("x")
        plt.ylabel("u(x, t)")
        plt.grid()
        plt.legend()
        plt.pause(0.01)

plt.show()