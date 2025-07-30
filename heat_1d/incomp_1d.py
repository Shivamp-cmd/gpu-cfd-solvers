import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 400
L = 1.0
dx = L / nx
dt = 0.0005
nt = 400  # total time steps

gamma = 1.4

# Grid
x = cp.linspace(0, L, nx)

# Initial Conditions (Sod shock tube)
rho = cp.where(x < 0.5, 1.0, 0.125)
u   = cp.zeros_like(x)
p   = cp.where(x < 0.5, 1.0, 0.1)

# Conserved Variables
E = p / (gamma - 1.0) + 0.5 * rho * u**2
U = cp.stack([rho, rho * u, E])  # Shape: (3, nx)

def compute_flux(U):
    rho = U[0]
    mom = U[1]
    E   = U[2]

    u = mom / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * u**2)

    F1 = mom
    F2 = mom * u + p
    F3 = u * (E + p)
    return cp.stack([F1, F2, F3])

def apply_boundary(U):
    # Neumann BC (zero-gradient)
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]
    return U

# Main loop
for n in range(nt):
    F = compute_flux(U)

    # Lax-Friedrichs scheme (simple, undergrad-friendly)
    U_next = 0.5 * (U[:, 2:] + U[:, :-2]) - dt / (2*dx) * (F[:, 2:] - F[:, :-2])

    # Pad with boundary conditions
    U_new = cp.zeros_like(U)
    U_new[:, 1:-1] = U_next
    U = apply_boundary(U_new)

    if n % 100 == 0:
        print(f"Step {n}/{nt}")

# Extract final physical variables
rho = U[0]
mom = U[1]
E   = U[2]
u = mom / rho
p = (gamma - 1.0) * (E - 0.5 * rho * u**2)

# Plotting
x_cpu = cp.asnumpy(x)
rho_cpu = cp.asnumpy(rho)
u_cpu = cp.asnumpy(u)
p_cpu = cp.asnumpy(p)

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.plot(x_cpu, rho_cpu, label='Density')
plt.title('Density')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x_cpu, u_cpu, label='Velocity')
plt.title('Velocity')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x_cpu, p_cpu, label='Pressure')
plt.title('Pressure')
plt.grid(True)

plt.suptitle("1D Shock Tube (GPU - CuPy, Lax-Friedrichs)")
plt.tight_layout()
plt.show()
