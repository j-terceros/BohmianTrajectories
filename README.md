Aquí tienes el archivo `README.md` corregido y formateado correctamente en Markdown:

````markdown
# 2D Bohmian Trajectory Simulation in Julia

This repository contains a Julia code that simulates the trajectory of a quantum particle using the Bohmian mechanics approach in a 2D harmonic potential. The simulation uses the Split-Step Fourier Method (SSFM) to evolve the wave function and interpolate the Bohmian velocities.

## Features

- **2D Bohmian trajectory**: Tracks the particle's position over time.
- **Coherent state initialization**: Starts with a Gaussian wave packet.
- **Split-Step Fourier Method (SSFM)**: Used to evolve the wave function in time.
- **Bohmian velocity calculation**: Uses FFT and interpolation to compute the velocity field.
- **Interactive plot**: Visualizes the trajectory and wave function probability density.

## Requirements

- Julia (v1.6 or later)
- `GLMakie` for visualization
- `FFTW` for FFT
- `LinearAlgebra` for math operations
- `Interpolations` for interpolation

To install the required packages, run the following commands in Julia:

```julia
using Pkg
Pkg.add("GLMakie")
Pkg.add("FFTW")
Pkg.add("LinearAlgebra")
Pkg.add("Interpolations")
````

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/bohmian-trajectory-2d.git
   cd bohmian-trajectory-2d
   ```

2. Install dependencies:

   ```julia
   using Pkg
   Pkg.instantiate()
   ```

3. Run the script:

   ```julia
   include("bohmian_trajectory.jl")
   ```

## Parameters

You can adjust the following parameters in the script to modify the simulation:

* `Nx`, `Ny`: Grid resolution (default: `256x256`)
* `Lx`, `Ly`: Size of the domain (default: `20.0` units)
* `ωx`, `ωy`: Frequencies of the potential (default: `6.0`, `7.0`)
* `tmax`, `dt`: Maximum time and time step (default: `100.0`, `0.01`)
* `a0x`, `a0y`: Initial wave packet amplitudes (default: `1.0`)

## Visualization

The script generates a plot that shows:

* The **Bohmian trajectory** of the particle (in red).
* The **initial** and **final** positions of the particle (green and orange dots).
* The **probability density** of the wave function (|ψ|²) as a heatmap.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.