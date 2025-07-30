import cupy as cp
import matplotlib.pyplot as plt

# Grid setup
nx, ny = 100, 100  # grid size
tol = 1e-5
max_iter = 10000

# Create L-shaped mask: upper-right corner removed
mask = cp.ones((nx, ny), dtype=cp.float32)
mask[nx//2:, ny//2:] = 0  # cut upper-right corner to make 90° bend

# Initialize temperature field
T = cp.zeros((nx, ny), dtype=cp.float32)
T_new = T.copy()

# Boundary condition: left wall hot (e.g. 100°C), others 0°C
T[:, 0] = 100.0
T_new[:, 0] = 100.0

# Ensure boundary is kept where mask == 1
fixed_mask = cp.zeros_like(mask)
fixed_mask[:, 0] = 1  # left wall fixed

# Main iteration loop
error = 1.0
it = 0

while error > tol and it < max_iter:
    T_prev = T.copy()

    # Apply Jacobi update only to interior L-shape
    T_new[1:-1, 1:-1] = 0.25 * (
        T[1:-1, :-2] + T[1:-1, 2:] +
        T[:-2, 1:-1] + T[2:, 1:-1]
    )

    # Apply mask to cut away invalid (non-pipe) region
    T_new = T_new * mask + T * (1 - mask)

    # Restore fixed boundaries
    T_new = T_new * (1 - fixed_mask) + T * fixed_mask

    error = cp.max(cp.abs(T_new - T))
    T[:] = T_new[:]
    it += 1

print(f"Converged in {it} iterations.")

# Plot
plt.imshow(cp.asnumpy(T), cmap='inferno', origin='lower')
plt.colorbar(label='Temperature (°C)')
plt.title('2D Heat Conduction in a 90° Bent Pipe')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(False)
plt.show()





