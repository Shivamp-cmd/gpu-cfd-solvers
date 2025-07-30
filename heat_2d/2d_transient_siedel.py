import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation 

# --- Parameters ---
V = 256        # Grid dimension (V x V)
L = 1.0        # Physical length of the domain (e.g., 1.0 meter)
dx = L / (V - 1); dy = L / (V - 1) # Spatial step size
alpha = 0.01   # Thermal diffusivity (e.g., m^2/s)
CFL_heat = 0.2 # Stability criterion for explicit method (must be <= 0.25 for 2D)
dt = CFL_heat * dx**2 / alpha # Time step size

Nt = 1000      #
save_interval = 20 

# --- Initialization ---
T = cp.zeros((V, V), dtype=cp.float64)
T[0, :] = 120.0 # Initial hot edge (top boundary, appears at bottom in plot)
T_prev = T.copy()
T_next = cp.zeros_like(T)

T_snapshots_gpu = []
time_stamps = []

# Store the initial state
T_snapshots_gpu.append(T.copy())
time_stamps.append(0.0)

# --- Transient Heat Conduction Loop ---
print("Starting 2D Transient Heat Conduction Simulation...")
for n in range(1, Nt + 1):
    # Explicit Finite Difference Update Rule for interior points
    T_next[1:-1, 1:-1] = T[1:-1, 1:-1] + CFL_heat * (
        T[1:-1, :-2] + T[1:-1, 2:] + T[:-2, 1:-1] + T[2:, 1:-1] - 4 * T[1:-1, 1:-1]
    )

    # --- Boundary Conditions (Dirichlet) ---
    T_next[0, :] = 120.0 # Top boundary (heated)
    T_next[-1, :] = 0.0  # Bottom boundary (cold)
    T_next[:, 0] = 0.0   # Left boundary (cold)
    T_next[:, -1] = 0.0  # Right boundary (cold)

    # --- NaN/Inf check ---
    if cp.any(cp.isnan(T_next)) or cp.any(cp.isinf(T_next)):
        print(f" Simulation diverged at time step {n}. Stopping.")
        break

    # Shift time steps
    T_prev[:] = T[:]
    T[:] = T_next[:]

    # Save snapshot at specified intervals
    if n % save_interval == 0 or n == Nt: # Also save the very last frame
        T_snapshots_gpu.append(T.copy())
        time_stamps.append(n * dt)

print(f"Simulation complete. Saved {len(T_snapshots_gpu)} snapshots.")

# --- Post-processing and Animation ---

# Convert all stored CuPy snapshots to NumPy arrays for Matplotlib
T_snapshots_cpu = [cp.asnumpy(snap) for snap in T_snapshots_gpu]

fig, ax = plt.subplots(figsize=(8, 6))
# Create an initial imshow plot
im = ax.imshow(T_snapshots_cpu[0], cmap='hot', origin='lower', aspect='auto',
               vmin=0, vmax=120) # Set fixed color scale for consistency

# Add a color bar
cbar = fig.colorbar(im, ax=ax, label='Temperature (°C)')

# Set plot title and labels
ax.set_title(f"2D Transient Heat Conduction at t = {time_stamps[0]:.3f}s")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

# Text element to display current time
time_text = ax.text(0.02, 0.95, '', color='white', transform=ax.transAxes,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))


# Animation update function
def animate(i):
    im.set_array(T_snapshots_cpu[i])
    ax.set_title(f"2D Transient Heat Conduction at t = {time_stamps[i]:.3f}s")
    time_text.set_text(f'Time: {time_stamps[i]:.3f}s')
    return [im, time_text] # Return a list of artists that were modified

# Create the animation
ani = animation.FuncAnimation(
    fig, animate, frames=len(T_snapshots_cpu), interval=50, blit=True
)

plt.tight_layout()
plt.show()
print("Animation displayed.")
