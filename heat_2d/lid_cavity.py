import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 128, 128
dx, dy = Lx/Nx, Ly/Ny
dt = 0.001
nt = 500  # total time steps
Re = 100.0
U_top = 1.0
nu = U_top * Lx / Re

# Staggered grid shapes
u = cp.zeros((Nx+1, Ny))      # u at vertical faces
v = cp.zeros((Nx, Ny+1))      # v at horizontal faces
p = cp.zeros((Nx, Ny))        # pressure at centers

u_star = u.copy()
v_star = v.copy()
b = cp.zeros((Nx, Ny))        # RHS of pressure Poisson equation

# Helper function: build divergence of tentative velocity (∇·u*)
def compute_divergence(u, v, dx, dy):
    du_dx = (u[1:, :] - u[:-1, :]) / dx
    dv_dy = (v[:, 1:] - v[:, :-1]) / dy
    return du_dx + dv_dy

# Pressure Poisson Solver (Jacobi)
def poisson_pressure(p, b, dx, dy, nit=100):
    pn = p.copy()
    for _ in range(nit):
        pn[1:-1, 1:-1] = ((dy**2 * (p[2:, 1:-1] + p[:-2, 1:-1]) +
                           dx**2 * (p[1:-1, 2:] + p[1:-1, :-2]) -
                           dx**2 * dy**2 * b[1:-1, 1:-1]) /
                          (2.0 * (dx**2 + dy**2)))
        p = pn.copy()
    return p

# Boundary conditions
def apply_bc(u, v):
    # u BCs
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = U_top  # top lid
    # v BCs
    v[:, 0] = 0
    v[:, -1] = 0
    v[0, :] = 0
    v[-1, :] = 0
    return u, v

# Main time-stepping loop
for n in range(nt):
    u, v = apply_bc(u, v)

    # Compute tentative u*, v*
    u_star[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * (
    - (u[2:, 1:-1] - u[:-2, 1:-1]) * u[1:-1, 1:-1] / (2*dx)
    + nu * (
        (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 +
        (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    )
)


    v_star[1:-1, 1:-1] = v[1:-1, 1:-1] + dt * (
    - (v[1:-1, 2:] - v[1:-1, :-2]) * v[1:-1, 1:-1] / (2*dy)
    + nu * (
        (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dx**2 +
        (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dy**2
    )
)


    # Compute RHS for Poisson
    b = (1/dt) * compute_divergence(u_star, v_star, dx, dy)

    # Solve pressure
    p = poisson_pressure(p, b, dx, dy, nit=50)

    # Correct velocities
    u[1:-1, :] = u_star[1:-1, :] - dt * (p[1:, :] - p[:-1, :]) / dx
    v[:, 1:-1] = v_star[:, 1:-1] - dt * (p[:, 1:] - p[:, :-1]) / dy

    # Apply BCs after correction
    u, v = apply_bc(u, v)

    if n % 50 == 0:
        print(f"Step {n}/{nt}")

# Convert GPU arrays to CPU for plotting
u_np = cp.asnumpy(u)
v_np = cp.asnumpy(v)
p_np = cp.asnumpy(p)

# Interpolate to cell centers (for plotting)
u_center = 0.5 * (u_np[:-1, :] + u_np[1:, :])
v_center = 0.5 * (v_np[:, :-1] + v_np[:, 1:])

# Create meshgrid for plotting
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Create figure
plt.figure(figsize=(14, 6))

# --- 1. Velocity Field (Quiver) ---
plt.subplot(1, 3, 1)
plt.quiver(X, Y, u_center.T, v_center.T, scale=30, color='black')
plt.title("Velocity Field (Quiver)")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal')
plt.grid(True)

# --- 2. Streamlines ---
plt.subplot(1, 3, 2)
speed = np.sqrt(u_center.T**2 + v_center.T**2)
strm = plt.streamplot(X, Y, u_center.T, v_center.T, color=speed, cmap='viridis', density=2)
plt.colorbar(strm.lines, label="Speed")
plt.title("Streamlines")
plt.xlabel("X")
plt.axis('equal')
plt.grid(True)

# --- 3. Pressure Contours ---
plt.subplot(1, 3, 3)
cont = plt.contourf(X, Y, p_np.T, levels=50, cmap='coolwarm')
plt.colorbar(cont, label="Pressure")
plt.title("Pressure Contours")
plt.xlabel("X")
plt.axis('equal')
plt.grid(True)

plt.suptitle("Lid-Driven Cavity Flow Visualization", fontsize=16)
plt.tight_layout()
plt.show()



