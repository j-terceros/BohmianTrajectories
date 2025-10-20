# Bohmian Trajectories - Two-Script Workflow

This repository uses a **two-stage workflow** to efficiently compute and visualize Bohmian trajectories:

1. **Stage 1**: `videos_traj.jl` - Runs expensive GPU simulation and saves ψ(x,y,t)
2. **Stage 2**: `compute_trajectories.jl` - Creates videos showing trajectories on wavefunction evolution

## Why This Approach?

**Advantages:**
- ✅ **100-200x faster** simulation (no CPU-GPU sync every timestep)
- ✅ Create **stunning visualizations** with phase and density heatmaps
- ✅ Compute **multiple trajectories** from one simulation
- ✅ Try **different integration methods** without re-running simulation
- ✅ Test **different interpolation schemes** on the same data
- ✅ Add **new initial positions** anytime without re-simulating
- ✅ Much **simpler, cleaner code** in each stage

**Disadvantage:**
- ⚠️ Requires disk space for saved ψ data (~100-500 MB per run)

---

## Quick Start

```julia
# Step 1: Run simulation
include("videos_traj.jl")
main()
# → Creates: psi_evolution_20251020_165432.jld2

# Step 2: Edit compute_trajectories.jl
#    Change jld2_file to match your output filename

# Step 3: Create video
include("compute_trajectories.jl")
main()
# → Creates: single_trajectory.mp4
```

**Result**: A video showing the wavefunction phase and density with a Bohmian particle trajectory!

---

## Stage 1: Run the Simulation

### File: `videos_traj.jl`

**Purpose**: Evolve the quantum wavefunction on GPU and save periodic snapshots.

### Usage:

```julia
include("videos_traj.jl")
main()
```

### Key Parameters (edit in `main()` function):

```julia
Nx, Ny = 256, 256          # Grid resolution
dt = 0.00001               # Time step
tmax = 100.0               # Total simulation time
save_every = 40            # Save ψ every 40 timesteps
```

### Output:

- **File**: `psi_evolution_YYYYMMDD_HHMMSS.jld2`
- **Contents**:
  - `meta/*` - All parameters needed for post-processing
  - `snapshots/it_NNNNNN/psi` - Complex wavefunction at each saved timestep
  - `snapshots/it_NNNNNN/t` - Time value

---

## Stage 2: Create Trajectory Visualizations

### File: `compute_trajectories.jl`

**Purpose**: Load saved ψ data and create videos showing Bohmian trajectories overlaid on wavefunction evolution.

### Usage:

1. **Edit** the `main()` function to specify:
   - Your JLD2 file name
   - Initial positions for particles
   - Integration method and video parameters

2. **Run**:
```julia
include("compute_trajectories.jl")
main()
```

### Example 1: Single Trajectory Video

```julia
function main()
    jld2_file = "psi_evolution_20251020_123456.jld2"
    
    create_trajectory_video(
        jld2_file,
        -2.0, 2.0,              # Initial position (x0, y0)
        method=:rk4,            # Integration method
        substeps=10,            # Substeps between snapshots
        output_file="single_trajectory.mp4",
        framerate=30
    )
end
```

### Example 2: Multiple Trajectories Video

```julia
function main()
    jld2_file = "psi_evolution_20251020_123456.jld2"
    
    initial_positions = [
        (-2.0, 2.0),
        (-1.0, 1.0),
        (0.0, 0.0),
        (1.0, -1.0),
    ]
    
    create_multiple_trajectory_video(
        jld2_file,
        initial_positions,
        method=:rk4,
        substeps=10,
        output_file="multiple_trajectories.mp4",
        framerate=30
    )
end
```

### Output:

- **MP4 Video** with dual-panel visualization:
  - **Left Panel**: Phase `angle(ψ)` with trajectory overlay
  - **Right Panel**: Density `|ψ|²` with trajectory overlay
- **Features**:
  - Real-time time counter
  - Trajectory history (red path)
  - Current particle position (yellow marker)
  - Color-coded multiple trajectories
  - Proper colorbars and labels

### Optional: Export to CSV

If you need trajectory data files instead of videos:

```julia
# Compute trajectories and save to CSV
trajectories = compute_multiple_trajectories(
    jld2_file, 
    [(-2.0, 2.0), (-1.0, 1.0)],
    method=:rk4,
    substeps=10
)

for (i, traj) in enumerate(trajectories)
    CSV.write("trajectory_$i.csv", traj)
end
```

**CSV Columns**: `time`, `x`, `y`, `vx`, `vy`

---

## Integration Methods

The post-processing script supports three integration methods:

### `:euler` (Fastest, Least Accurate)
- Simple forward Euler: `x_{n+1} = x_n + dt * v_n`
- Good for quick tests

### `:rk2` (Moderate Speed & Accuracy)
- 2nd-order Runge-Kutta (midpoint method)
- Better accuracy than Euler

### `:rk4` (Slowest, Most Accurate)
- 4th-order Runge-Kutta
- **Recommended for final results**

### Substeps Parameter

Since ψ is only saved periodically, you can increase `substeps` to integrate more smoothly between snapshots:

- `substeps=1`: One integration step per snapshot (fastest, least smooth)
- `substeps=10`: 10 integration steps per snapshot (recommended)
- `substeps=100`: Very smooth, but slower

---

## Video Customization

### Change Colormaps

Edit the video creation functions to customize colors:

```julia
# For phase visualization (left panel)
colormap=:twilight          # Default (excellent for phase)
colormap=:cyclic_mrybm_35_75_c68_n256  # Alternative cyclic colormap
colormap=:hsv               # HSV color wheel

# For density visualization (right panel)
colormap=:viridis           # Default (perceptually uniform)
colormap=:plasma            # Purple-orange
colormap=:inferno           # Black-red-yellow
colormap=:cividis           # Colorblind-friendly
```

### Adjust Visual Elements

```julia
# Trajectory appearance
linewidth=2                 # Thickness of trajectory path
color=:red                  # Color of trajectory line
markersize=15               # Size of current particle marker

# Video parameters
framerate=30                # Frames per second (higher = smoother)
resolution=(1600, 700)      # Video resolution in pixels
```

### Change Video Format

GLMakie supports multiple formats:

```julia
output_file="trajectory.mp4"   # H.264 MP4 (default)
output_file="trajectory.mkv"   # Matroska container
output_file="trajectory.webm"  # WebM format
```

### Publication-Quality Videos

For papers/presentations, consider:

```julia
# High resolution, smooth animation
create_trajectory_video(
    jld2_file, x0, y0,
    method=:rk4,
    substeps=20,              # Very smooth integration
    output_file="paper_figure.mp4",
    framerate=60,             # Silky smooth playback
)
```

Then edit `compute_trajectories.jl` to set:
```julia
resolution=(2400, 1000)       # High-res for presentations
fontsize=28                   # Larger fonts for readability
linewidth=3                   # Thicker trajectory for visibility
```

### Tips for Great Videos

- **For talks**: 30 fps at 1600×700 is perfect
- **For papers**: 60 fps at 2400×1000 for supplementary materials  
- **For web**: 24 fps at 1200×500 for smaller file sizes
- **Phase colormap**: `:twilight` is best for ψ phase (cyclic at ±π)
- **Density colormap**: `:viridis` or `:plasma` (perceptually uniform)

---

## Advanced Usage

### Compute Single Trajectory (CSV Output)

```julia
df = compute_trajectory("psi_evolution_20251020_123456.jld2", -2.0, 2.0, 
                        method=:rk4, substeps=10)
CSV.write("my_trajectory.csv", df)
```

### Custom Interpolation

Modify the `bilinear_interpolation` function to try:
- Bicubic interpolation
- Spectral interpolation
- Higher-order methods

### Different Velocity Formulations

Modify `compute_bohmian_velocity_fft` to implement:
- Different regularization schemes (change `ε`)
- Filtered velocity fields (apply smoothing)
- Alternative quantum potential formulations

### Add Custom Overlays

Edit the video functions to add:
- Streamlines of velocity field
- Contour lines of |ψ|²
- Classical trajectory comparisons
- Quantum potential visualization

---

## File Structure

```
BohmianTrajectories/
├── videos_traj.jl              # Stage 1: GPU simulation
├── compute_trajectories.jl      # Stage 2: Video creation & analysis
├── README_WORKFLOW.md           # This file
│
├── psi_evolution_*.jld2         # Output from Stage 1 (large files)
├── *.mp4                        # Output from Stage 2 (trajectory videos)
└── trajectory_*.csv             # Optional CSV exports
```

---

## Performance Tips

### Stage 1 (Simulation):
- Increase `save_every` to save disk space and I/O time
- Use smaller grids (`Nx`, `Ny`) for testing
- Monitor GPU memory usage with `nvidia-smi`

### Stage 2 (Video Creation):
- Use `:euler` with `substeps=1` for quick preview videos
- Use `:rk4` with `substeps=10` for final publication videos
- Lower `framerate` (e.g., 15 fps) for faster rendering
- Reduce resolution to (1200, 500) for draft videos
- Videos render at ~10-50 frames/second depending on grid size

---

## Memory Requirements

### GPU Memory (Stage 1):
- ψ array: `Nx × Ny × 16 bytes` (ComplexF64)
- Operators: ~6 arrays of size `Nx × Ny × 16 bytes`
- **Example**: 256×256 grid ≈ 6 MB GPU memory

### Disk Space (Stage 1 Output):
- Per snapshot: `Nx × Ny × 16 bytes`
- **Example**: 256×256 grid, 2500 snapshots ≈ 160 MB

### RAM (Stage 2):
- Loads one snapshot at a time
- **Example**: 256×256 grid ≈ 1 MB per snapshot

---

## Troubleshooting

### "Out of GPU Memory"
- Reduce `Nx`, `Ny`
- Check for memory leaks in custom code

### "JLD2 file too large"
- Increase `save_every` (trade temporal resolution for space)
- Reduce `tmax`
- Use compression: `jldopen(file, "w"; compress=true)`

### "Trajectories look wrong"
- Check initial positions are within domain
- Increase `substeps` for smoother integration
- Try different integration methods
- Verify ψ data is correct by plotting first frame

### "Video rendering is slow"
- Reduce `framerate` (15-20 fps is often sufficient)
- Reduce figure resolution
- Use `:euler` with `substeps=1` for draft previews
- Consider reducing grid size in simulation

### "Video file is huge"
- Lower `framerate` (30 fps → 15 fps halves file size)
- Reduce resolution (1600×700 → 1200×500)
- Use more compression: some players support `.webm` better

### "GLMakie display issues"
- Make sure you have OpenGL 3.3+ support
- Try setting `GLMakie.activate!(; ssao=false)`
- Update GPU drivers
- On remote systems, use `WGLMakie` instead of `GLMakie`

### "Particle disappears from frame"
- Check if particle left the domain
- Enable periodic boundaries (uncomment wrap code in functions)
- Adjust initial position to stay within interesting region

---

## Citation

If you use this code in your research, please cite:

```
[Add your citation here]
```

---

## License

[Add your license here]

