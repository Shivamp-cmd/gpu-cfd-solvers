import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 15.0
Nx = 2000
dx = L / (Nx - 1)
alpha = 1e-4
dt = 0.4 * dx**2 / alpha
Nt = 2000  # number of time steps
r = alpha * dt / dx**2

# Grid
x = cp.linspace(0, L, Nx)

# Initial condition
T = cp.zeros(Nx)
T[Nx // 2] = 100
T_new = T.copy()

# Store all time steps
T_record = cp.zeros((Nt, Nx))

# Time loop
for n in range(Nt):
    T_new[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[:-2])
    T_new[0] = 0
    T_new[-1] = 0
    T[:] = T_new
    T_record[n] = T

# Convert to CPU for plotting
T_record_cpu = cp.asnumpy(T_record)

# Build time and space axes
x_cpu = np.linspace(0, L, Nx)
t_cpu = np.linspace(0, Nt * dt, Nt)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(
    T_record_cpu,
    extent=[0, L, 0, Nt * dt],
    aspect='auto',
    cmap='hot',
    origin='lower'
)
plt.colorbar(label="Temperature (°C)")
plt.xlabel("Rod Position (m)")
plt.ylabel("Time (s)")
plt.title("1D Unsteady Heat Conduction Over Time")
plt.tight_layout()
plt.show()
