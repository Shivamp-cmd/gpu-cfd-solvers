import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Nx, Ny = 256, 256
Lx, Ly = 1.0, 1.0
dx, dy = Lx/Nx, Ly/Ny

# Solver parameters
omega = 1.4          # safer relaxation factor
tol = 1e-5
max_iter = 5000

# CuPy arrays with ghost cells
phi = cp.zeros((Nx+2, Ny+2))
b = cp.zeros_like(phi)

# Safe source term: mild delta function
b[Nx//2, Ny//2] = 10.0

# SOR Solver
def poisson_sor(phi, b, dx, dy, omega, tol, max_iter):
    for it in range(max_iter):
        phi_old = phi.copy()

        # SOR update
        phi[1:-1, 1:-1] = (1 - omega) * phi[1:-1, 1:-1] + omega * 0.25 * (
            phi[2:, 1:-1] + phi[:-2, 1:-1] +
            phi[1:-1, 2:] + phi[1:-1, :-2] -
            dx**2 * b[1:-1, 1:-1]
        )

        # Dirichlet BCs
        phi[0, :] = 0
        phi[-1, :] = 0
        phi[:, 0] = 0
        phi[:, -1] = 0

        # NaN/Inf check
        if cp.any(cp.isnan(phi)) or cp.any(cp.isinf(phi)):
            print(f"Diverged at iteration {it}")
            break

        # Convergence check
        diff = cp.max(cp.abs(phi - phi_old))
        if it % 100 == 0 or diff < tol:
            print(f"Iteration {it}: residual = {diff:.2e}")
        if diff < tol:
            break

    return phi

# Run the solver
phi = poisson_sor(phi, b, dx, dy, omega, tol, max_iter)

# Clip final result (just in case)
phi = cp.clip(phi, -1e3, 1e3)

# Transfer to CPU for plotting
phi_np = cp.asnumpy(phi)

# Meshgrid for plotting
x = np.linspace(0, Lx, Nx+2)
y = np.linspace(0, Ly, Ny+2)
X, Y = np.meshgrid(x, y)

# Plot
plt.figure(figsize=(8, 6))
cont = plt.contourf(X, Y, phi_np.T, levels=50, cmap='plasma')
plt.colorbar(cont, label='ϕ')
plt.title("Stable 2D Poisson Solver (GPU - CuPy + SOR)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.tight_layout()
plt.show()

