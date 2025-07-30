import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

Nx,Ny= 1000, 1000
Lx,Ly= 90, 50
dx, dy= Lx/Nx , Ly/Ny


alpha = 0.01
dt =min(dx,dy)**2/ (4*alpha)
max_time=100

num_step = int(max_time /dt)
T=cp.zeros((Nx,Ny))
T_new = cp.zeros_like(T)
T[0, :]= 100
T[-1,:]=0
T[:,0]= cp.sin(cp.pi*cp.linspace(0,1,Ny))*100
T[:,-1]= 0.5*(T[:,-2] + T[:,0])
X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
def solve_transient_heat_conduction(T, T_new, alpha, dt, dx, dy, num_steps):
  for step in range (num_step):
      T_new=T.copy()
      T_new[1:-1,1:-1] = T[1:-1,1:-1] \
            + alpha*dt * (
            (T[2:,1:-1] - 2*T[1:-1,1:-1] + T[:-2,1:-1]) / dx**2
           + (T[1:-1,2:] - 2*T[1:-1,1:-1] + T[1:-1,:-2]) / dy**2
          )
      T_new[0, :] = 100
      T_new[-1, :] = 0
      T_new[:, 0] = cp.sin(cp.pi * cp.linspace(0, 1, Ny)) * 100
      T_new[:, -1] = 0.5 * (T_new[:, -2] + T_new[:, 0])
      T[:] = T_new
      if step % 50 == 0 or step == num_steps - 1:
            plt.figure(figsize=(8, 6))
            plt.contourf(X, Y, cp.asnumpy(T), 50, cmap='hot')
            plt.colorbar(label="Temperature (°C)")
            plt.title(f"Temperature at t={step*dt:.4f} s")
            plt.xlabel("X direction")
            plt.ylabel("Y direction")
            plt.tight_layout()
            plt.show()


  return T

# Solve the transient heat conduction problem
T_solution = solve_transient_heat_conduction(T, T_new, alpha, dt, dx, dy, num_step)