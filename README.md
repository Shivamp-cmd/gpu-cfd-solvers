name: GPU Numerical Solvers
description: |
  📌 Educational GPU-accelerated numerical solvers for fundamental PDEs — practical examples for students and hobbyists.
  These codes are simple, transparent, and ideal for learning core numerical methods, not for production CFD/FEA pipelines.
  Developed using Python, CuPy (GPU backend), and Matplotlib for visualization.

author: Your Name
topics:
  - CFD
  - Heat Transfer
  - Wave Equation
  - Poisson Equation
  - Compressible Flow
  - GPU Computing
  - CuPy
  - Numerical Methods
  - Educational Examples

projects:
  - name: 2D Poisson Solver (SOR)
    description: |
      Solves the 2D Poisson equation using the Successive Over-Relaxation method.
      GPU-accelerated with CuPy, Dirichlet BCs, mild point source.
  - name: 1D Wave Equation
    description: |
      Simulates propagation of a 1D wave using explicit finite differences.
      Initial Gaussian pulse, fixed BCs, CFL condition enforced.
  - name: 2D Transient Heat Conduction
    description: |
      Explicit FTCS scheme for 2D heat conduction in a square plate.
      Top edge heated, other edges cold, GPU time stepping, animated result.
  - name: 2D Heat Conduction (Mixed BCs)
    description: |
      Similar to the basic 2D solver, but mixes Dirichlet and Neumann BCs.
      Left edge with sine temperature, right edge insulated.
  - name: 1D Unsteady Heat Conduction in a Rod
    description: |
      FTCS for 1D rod with point source initial condition.
      Ends held at zero temp, result visualized as spacetime plot.
  - name: 1D Compressible Flow (Sod Shock Tube)
    description: |
      Classic shock tube solved using Lax-Friedrichs scheme.
      Demonstrates shock wave and rarefaction in 1D Euler equations.
      Density, velocity, and pressure visualized.

requirements:
  - Python >= 3.8
  - cupy >= 12.0
  - numpy >= 1.21
  - matplotlib >= 3.5

usage: |
  1. Clone this repo.
  2. Install dependencies: pip install -r requirements.txt
  3. Run any solver: python poisson_solver.py
  4. Inspect results, tweak parameters, and learn!

note: |
  These solvers are designed for educational demonstrations.
  They are simplified compared to industrial-strength CFD or FEA workflows.
  Feel free to adapt, expand, or port them to other frameworks.

repository:
  homepage: https://github.com/YourUsername/GPU_Numerical_Solvers
  issues: https://github.com/YourUsername/GPU_Numerical_Solvers/issues
