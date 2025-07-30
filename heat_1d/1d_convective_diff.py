import cupy as cp
import matplotlib.pyplot as plt
import numpy as np # For plotting on CPU

# --- Parameters ---
L = 1.0        # Domain length (e.g., 1.0 meter)
Nx = 500       # Number of spatial grid points
dx = L / (Nx - 1) # Spatial step size
D = 0.001      
u_vel = 0.1    
CFL_factor = 0.4
dt_diff = dx**2 / (2 * D) if D > 0 else cp.inf
dt_conv = dx / abs(u_vel) if u_vel != 0 else cp.inf
dt = min(dt_diff, dt_conv) * CFL_factor
Nt = 2000      
plot_interval = 50 

# --- Initialization ---
C = cp.zeros(Nx, dtype=cp.float64)
C_next = cp.zeros_like(C)

# Spatial grid (on GPU)
x = cp.linspace(0, L, Nx)
# --- Initial Condition (Gaussian pulse at t=0) ---
C[:] = cp.exp(-1000 * (x - 0.5)**2)
# --- Setup for Live Plotting ---
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(cp.asnumpy(x), cp.asnumpy(C), lw=2) 
ax.set_ylim(-0.2, 1.2) 
ax.set_title(f"1D Convection-Diffusion at t = {0.0:.3f}s")
ax.set_xlabel("X Position")
ax.set_ylabel("Concentration (C)")
ax.grid(True)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes) 
plt.tight_layout()

# --- Convection-Diffusion Loop ---
print(f"Starting 1D Convection-Diffusion Simulation (dt={dt:.6f}s)...")
for n in range(1, Nt + 1):
   
    diffusion_term = D * (C[2:] - 2 * C[1:-1] + C[:-2]) / dx**2
    convection_term = -u_vel * (C[2:] - C[:-2]) / (2 * dx)
    # Combine terms for C_next
    C_next[1:-1] = C[1:-1] + dt * (diffusion_term + convection_term)

    # --- Boundary Conditions (Dirichlet) ---
    C_next[-1] = 0.0
    # --- NaN/Inf check ---
    if cp.any(cp.isnan(C_next)) or cp.any(cp.isinf(C_next)):
        print(f" Simulation diverged at time step {n}. Stopping.")
        break
    C[:] = C_next[:] 
    # --- Live Plotting ---
    if n % plot_interval == 0 or n == Nt:
        current_time = n * dt
        line.set_ydata(cp.asnumpy(C)) # Update the line data
        ax.set_title(f"1D Convection-Diffusion at t = {current_time:.3f}s")
        time_text.set_text(f'Time: {current_time:.3f}s')
        plt.draw() # Redraw the canvas
        plt.pause(0.01) #

print(f"Simulation complete. Final time: {Nt * dt:.3f}s.")
plt.show() 
print("Animation displayed.")