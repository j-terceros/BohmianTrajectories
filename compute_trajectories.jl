using JLD2
using FFTW
using CSV, DataFrames
using Printf
using LinearAlgebra
using GLMakie

"""
    compute_bohmian_velocity_fft(ψ, kx, ky)

Compute Bohmian velocity field from wavefunction using FFT derivatives.
Returns velocity fields vx, vy.
"""
function compute_bohmian_velocity_fft(ψ::Matrix{ComplexF64}, kx::Vector{Float64}, ky::Vector{Float64})
    Nx, Ny = size(ψ)
    
    # Fourier transform
    ψ_hat = fft(ψ)
    
    # Spectral derivatives
    KX = repeat(kx, 1, Ny)
    KY = repeat(ky', Nx, 1)
    
    dψdx_hat = (1im) .* KX .* ψ_hat
    dψdy_hat = (1im) .* KY .* ψ_hat
    
    dψdx = ifft(dψdx_hat)
    dψdy = ifft(dψdy_hat)
    
    # Bohmian velocity: v = ℏ/m * Im(∇ψ/ψ)
    ε = 1e-15
    vx = imag.(dψdx ./ (ψ .+ ε))
    vy = imag.(dψdy ./ (ψ .+ ε))
    
    return vx, vy
end

"""
    bilinear_interpolation(field, x0, dx, y0, dy, xp, yp)

Bilinear interpolation of a 2D field at position (xp, yp).
"""
function bilinear_interpolation(field::Matrix{Float64}, x0::Float64, dx::Float64, 
                                 y0::Float64, dy::Float64, xp::Float64, yp::Float64)
    i = (xp - x0) / dx
    j = (yp - y0) / dy
    ix = Int(floor(i)) + 1
    jx = Int(floor(j)) + 1
    wx = i - floor(i)
    wy = j - floor(j)
    
    # Clamp indices to valid range
    Nx, Ny = size(field)
    ix = clamp(ix, 1, Nx-1)
    jx = clamp(jx, 1, Ny-1)
    
    # Bilinear weights
    v00 = field[ix, jx]
    v10 = field[ix+1, jx]
    v01 = field[ix, jx+1]
    v11 = field[ix+1, jx+1]
    
    return (1-wx)*(1-wy)*v00 + wx*(1-wy)*v10 + (1-wx)*wy*v01 + wx*wy*v11
end

"""
    compute_trajectory(jld2_file, x0, y0; method=:euler, substeps=1)

Compute a single Bohmian trajectory from saved ψ data.

Arguments:
- jld2_file: Path to the JLD2 file with saved ψ snapshots
- x0, y0: Initial particle position
- method: Integration method (:euler, :rk2, :rk4)
- substeps: Number of substeps between snapshots (for better accuracy)

Returns:
- DataFrame with columns: time, x, y, vx, vy
"""
function compute_trajectory(jld2_file::String, x0::Float64, y0::Float64; 
                            method::Symbol=:euler, substeps::Int=1)
    
    jldopen(jld2_file, "r") do f
        # Load metadata
        x = f["meta/x"]
        y = f["meta/y"]
        dx = f["meta/dx"]
        dy = f["meta/dy"]
        dt = f["meta/dt"]
        save_every = f["meta/save_every"]
        
        # Build frequency grids for FFT
        Nx, Ny = length(x), length(y)
        kx = fftfreq(Nx, 2π/dx)
        ky = fftfreq(Ny, 2π/dy)
        
        # Find all snapshot keys
        snapshot_keys = sort([k for k in keys(f["snapshots"])])
        n_snapshots = length(snapshot_keys)
        
        println("Processing $n_snapshots snapshots...")
        println("Initial position: ($x0, $y0)")
        println("Integration method: $method")
        println("Substeps per snapshot: $substeps")
        
        # Initialize trajectory arrays
        times = Float64[]
        xs = Float64[]
        ys = Float64[]
        vxs = Float64[]
        vys = Float64[]
        
        # Current particle position
        xp, yp = x0, y0
        
        # Time step for integration
        dt_integ = dt * save_every / substeps
        
        for i in 1:n_snapshots
            snap_key = snapshot_keys[i]
            t = f["snapshots/$snap_key/t"]
            ψ = f["snapshots/$snap_key/psi"]
            
            # Compute velocity field
            vx_field, vy_field = compute_bohmian_velocity_fft(ψ, kx, ky)
            
            # Sample velocity at particle position
            vx_here = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xp, yp)
            vy_here = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xp, yp)
            
            # Store current state
            push!(times, t)
            push!(xs, xp)
            push!(ys, yp)
            push!(vxs, vx_here)
            push!(vys, vy_here)
            
            # Integrate to next snapshot (unless last snapshot)
            if i < n_snapshots
                if method == :euler
                    # Simple Euler step
                    xp += dt_integ * substeps * vx_here
                    yp += dt_integ * substeps * vy_here
                    
                elseif method == :rk2
                    # RK2 (midpoint) - approximate
                    for _ in 1:substeps
                        k1x, k1y = vx_here, vy_here
                        xmid = xp + 0.5*dt_integ*k1x
                        ymid = yp + 0.5*dt_integ*k1y
                        k2x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xmid, ymid)
                        k2y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xmid, ymid)
                        xp += dt_integ * k2x
                        yp += dt_integ * k2y
                        vx_here, vy_here = k2x, k2y
                    end
                    
                elseif method == :rk4
                    # RK4 - approximate (velocity field constant between snapshots)
                    for _ in 1:substeps
                        k1x, k1y = vx_here, vy_here
                        
                        xmid = xp + 0.5*dt_integ*k1x
                        ymid = yp + 0.5*dt_integ*k1y
                        k2x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xmid, ymid)
                        k2y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xmid, ymid)
                        
                        xmid = xp + 0.5*dt_integ*k2x
                        ymid = yp + 0.5*dt_integ*k2y
                        k3x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xmid, ymid)
                        k3y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xmid, ymid)
                        
                        xend = xp + dt_integ*k3x
                        yend = yp + dt_integ*k3y
                        k4x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xend, yend)
                        k4y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xend, yend)
                        
                        xp += (dt_integ/6) * (k1x + 2*k2x + 2*k3x + k4x)
                        yp += (dt_integ/6) * (k1y + 2*k2y + 2*k3y + k4y)
                        
                        vx_here = k4x
                        vy_here = k4y
                    end
                end
                
                # Optional: wrap to periodic boundary
                # xp = mod(xp - x[1], x[end]-x[1]) + x[1]
                # yp = mod(yp - y[1], y[end]-y[1]) + y[1]
            end
            
            if i % 10 == 0
                println("  Processed snapshot $i/$n_snapshots")
            end
        end
        
        return DataFrame(time=times, x=xs, y=ys, vx=vxs, vy=vys)
    end
end

"""
    compute_multiple_trajectories(jld2_file, initial_positions; kwargs...)

Compute multiple Bohmian trajectories from saved ψ data.

Arguments:
- jld2_file: Path to the JLD2 file
- initial_positions: Vector of (x0, y0) tuples
- kwargs: Additional arguments passed to compute_trajectory

Returns:
- Vector of DataFrames, one per trajectory
"""
function compute_multiple_trajectories(jld2_file::String, 
                                        initial_positions::Vector{Tuple{Float64,Float64}};
                                        kwargs...)
    trajectories = DataFrame[]
    
    for (i, (x0, y0)) in enumerate(initial_positions)
        println("\n=== Computing trajectory $i/$(length(initial_positions)) ===")
        traj = compute_trajectory(jld2_file, x0, y0; kwargs...)
        push!(trajectories, traj)
    end
    
    return trajectories
end

"""
    create_trajectory_video(jld2_file, x0, y0; method=:rk4, substeps=1, 
                            output_file="trajectory_video.mp4", framerate=30)

Create an animated video showing the wavefunction evolution and Bohmian trajectory.

Arguments:
- jld2_file: Path to the JLD2 file with saved ψ snapshots
- x0, y0: Initial particle position
- method: Integration method (:euler, :rk2, :rk4)
- substeps: Number of substeps between snapshots
- output_file: Output video filename
- framerate: Video framerate (fps)

The video shows:
- Left panel: Phase angle(ψ) with trajectory overlay
- Right panel: Density |ψ|²
"""
function create_trajectory_video(jld2_file::String, x0::Float64, y0::Float64; 
                                 method::Symbol=:rk4, substeps::Int=1,
                                 output_file::String="trajectory_video.mp4",
                                 framerate::Int=30)
    
    jldopen(jld2_file, "r") do f
        # Load metadata
        x = f["meta/x"]
        y = f["meta/y"]
        dx = f["meta/dx"]
        dy = f["meta/dy"]
        dt = f["meta/dt"]
        save_every = f["meta/save_every"]
        
        # Build frequency grids for FFT
        Nx, Ny = length(x), length(y)
        kx = fftfreq(Nx, 2π/dx)
        ky = fftfreq(Ny, 2π/dy)
        
        # Find all snapshot keys
        snapshot_keys = sort([k for k in keys(f["snapshots"])])
        n_snapshots = length(snapshot_keys)
        
        println("Creating video with $n_snapshots frames...")
        println("Initial position: ($x0, $y0)")
        println("Integration method: $method")
        println("Substeps per snapshot: $substeps")
        
        # Arrays to store trajectory
        traj_x = Float64[x0]
        traj_y = Float64[y0]
        
        # Current particle position
        xp, yp = x0, y0
        
        # Time step for integration
        dt_integ = dt * save_every / substeps
        
        # Create figure
        fig = Figure(resolution=(1600, 700))
        
        # Create axes
        ax1 = Axis(fig[1, 1], 
                   xlabel="x", ylabel="y",
                   title="Phase: angle(ψ)",
                   aspect=DataAspect())
        ax2 = Axis(fig[1, 2], 
                   xlabel="x", ylabel="y",
                   title="Density: |ψ|²",
                   aspect=DataAspect())
        
        # Load first snapshot to initialize
        snap_key = snapshot_keys[1]
        ψ_first = f["snapshots/$snap_key/psi"]
        
        # Create observables for heatmaps
        phase_obs = Observable(angle.(ψ_first))
        density_obs = Observable(abs2.(ψ_first))
        
        # Create heatmaps
        hm1 = heatmap!(ax1, x, y, phase_obs, 
                      colormap=:twilight, 
                      colorrange=(-π, π))
        Colorbar(fig[1, 1, Right()], hm1, label="Phase (rad)")
        
        hm2 = heatmap!(ax2, x, y, density_obs, 
                      colormap=:viridis)
        Colorbar(fig[1, 2, Right()], hm2, label="Density")
        
        # Create observables for trajectory
        traj_x_obs = Observable(traj_x)
        traj_y_obs = Observable(traj_y)
        
        # Plot trajectory on both axes
        lines!(ax1, traj_x_obs, traj_y_obs, color=:red, linewidth=2, label="Trajectory")
        scatter!(ax1, traj_x_obs, traj_y_obs, color=:red, markersize=8)
        scatter!(ax1, [traj_x_obs[][end]], [traj_y_obs[][end]], 
                color=:yellow, markersize=15, marker='●', label="Particle")
        
        lines!(ax2, traj_x_obs, traj_y_obs, color=:red, linewidth=2)
        scatter!(ax2, traj_x_obs, traj_y_obs, color=:red, markersize=8)
        scatter!(ax2, [traj_x_obs[][end]], [traj_y_obs[][end]], 
                color=:yellow, markersize=15, marker='●')
        
        # Add time display
        time_text = Observable(@sprintf("t = %.4f", 0.0))
        Label(fig[0, :], time_text, fontsize=24, tellwidth=false)
        
        # Record video
        record(fig, output_file, 1:n_snapshots; framerate=framerate) do frame_idx
            snap_key = snapshot_keys[frame_idx]
            t = f["snapshots/$snap_key/t"]
            ψ = f["snapshots/$snap_key/psi"]
            
            # Update heatmaps
            phase_obs[] = angle.(ψ)
            density_obs[] = abs2.(ψ)
            
            # Update time display
            time_text[] = @sprintf("t = %.4f", t)
            
            # Compute trajectory for this frame (if not first frame)
            if frame_idx > 1
                # Get velocity field
                vx_field, vy_field = compute_bohmian_velocity_fft(ψ, kx, ky)
                
                # Sample velocity at particle position
                vx_here = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xp, yp)
                vy_here = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xp, yp)
                
                # Integrate to current position
                if method == :euler
                    xp += dt_integ * substeps * vx_here
                    yp += dt_integ * substeps * vy_here
                    
                elseif method == :rk2
                    for _ in 1:substeps
                        k1x, k1y = vx_here, vy_here
                        xmid = xp + 0.5*dt_integ*k1x
                        ymid = yp + 0.5*dt_integ*k1y
                        k2x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xmid, ymid)
                        k2y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xmid, ymid)
                        xp += dt_integ * k2x
                        yp += dt_integ * k2y
                        vx_here, vy_here = k2x, k2y
                    end
                    
                elseif method == :rk4
                    for _ in 1:substeps
                        k1x, k1y = vx_here, vy_here
                        
                        xmid = xp + 0.5*dt_integ*k1x
                        ymid = yp + 0.5*dt_integ*k1y
                        k2x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xmid, ymid)
                        k2y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xmid, ymid)
                        
                        xmid = xp + 0.5*dt_integ*k2x
                        ymid = yp + 0.5*dt_integ*k2y
                        k3x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xmid, ymid)
                        k3y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xmid, ymid)
                        
                        xend = xp + dt_integ*k3x
                        yend = yp + dt_integ*k3y
                        k4x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xend, yend)
                        k4y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xend, yend)
                        
                        xp += (dt_integ/6) * (k1x + 2*k2x + 2*k3x + k4x)
                        yp += (dt_integ/6) * (k1y + 2*k2y + 2*k3y + k4y)
                        
                        vx_here = k4x
                        vy_here = k4y
                    end
                end
                
                # Add current position to trajectory
                push!(traj_x, xp)
                push!(traj_y, yp)
                
                # Update trajectory observables
                traj_x_obs[] = copy(traj_x)
                traj_y_obs[] = copy(traj_y)
            end
            
            if frame_idx % 10 == 0
                println("  Rendered frame $frame_idx/$n_snapshots")
            end
        end
        
        println("\n✅ Video saved to: $output_file")
    end
end

"""
    create_multiple_trajectory_video(jld2_file, initial_positions; 
                                     method=:rk4, substeps=1,
                                     output_file="trajectories_video.mp4", 
                                     framerate=30)

Create an animated video showing multiple Bohmian trajectories simultaneously.

Arguments:
- jld2_file: Path to the JLD2 file with saved ψ snapshots
- initial_positions: Vector of (x0, y0) tuples
- method: Integration method (:euler, :rk2, :rk4)
- substeps: Number of substeps between snapshots
- output_file: Output video filename
- framerate: Video framerate (fps)

The video shows:
- Left panel: Phase angle(ψ) with all trajectories
- Right panel: Density |ψ|²
"""
function create_multiple_trajectory_video(jld2_file::String, 
                                          initial_positions::Vector{Tuple{Float64,Float64}}; 
                                          method::Symbol=:rk4, substeps::Int=1,
                                          output_file::String="trajectories_video.mp4",
                                          framerate::Int=30)
    
    jldopen(jld2_file, "r") do f
        # Load metadata
        x = f["meta/x"]
        y = f["meta/y"]
        dx = f["meta/dx"]
        dy = f["meta/dy"]
        dt = f["meta/dt"]
        save_every = f["meta/save_every"]
        
        # Build frequency grids for FFT
        Nx, Ny = length(x), length(y)
        kx = fftfreq(Nx, 2π/dx)
        ky = fftfreq(Ny, 2π/dy)
        
        # Find all snapshot keys
        snapshot_keys = sort([k for k in keys(f["snapshots"])])
        n_snapshots = length(snapshot_keys)
        n_trajectories = length(initial_positions)
        
        println("Creating video with $n_snapshots frames and $n_trajectories trajectories...")
        println("Integration method: $method")
        println("Substeps per snapshot: $substeps")
        
        # Initialize trajectories
        traj_xs = [[x0] for (x0, y0) in initial_positions]
        traj_ys = [[y0] for (x0, y0) in initial_positions]
        xps = [x0 for (x0, y0) in initial_positions]
        yps = [y0 for (x0, y0) in initial_positions]
        
        # Time step for integration
        dt_integ = dt * save_every / substeps
        
        # Create figure
        fig = Figure(resolution=(1600, 700))
        
        # Create axes
        ax1 = Axis(fig[1, 1], 
                   xlabel="x", ylabel="y",
                   title="Phase: angle(ψ)",
                   aspect=DataAspect())
        ax2 = Axis(fig[1, 2], 
                   xlabel="x", ylabel="y",
                   title="Density: |ψ|²",
                   aspect=DataAspect())
        
        # Load first snapshot
        snap_key = snapshot_keys[1]
        ψ_first = f["snapshots/$snap_key/psi"]
        
        # Create observables for heatmaps
        phase_obs = Observable(angle.(ψ_first))
        density_obs = Observable(abs2.(ψ_first))
        
        # Create heatmaps
        hm1 = heatmap!(ax1, x, y, phase_obs, 
                      colormap=:twilight, 
                      colorrange=(-π, π))
        Colorbar(fig[1, 1, Right()], hm1, label="Phase (rad)")
        
        hm2 = heatmap!(ax2, x, y, density_obs, 
                      colormap=:viridis)
        Colorbar(fig[1, 2, Right()], hm2, label="Density")
        
        # Create observables for trajectories
        traj_x_obs = [Observable(traj_xs[i]) for i in 1:n_trajectories]
        traj_y_obs = [Observable(traj_ys[i]) for i in 1:n_trajectories]
        
        # Color palette for different trajectories
        colors = Makie.wong_colors()
        
        # Plot all trajectories
        for i in 1:n_trajectories
            color_idx = mod1(i, length(colors))
            c = colors[color_idx]
            
            # Left panel
            lines!(ax1, traj_x_obs[i], traj_y_obs[i], color=c, linewidth=2)
            scatter!(ax1, traj_x_obs[i], traj_y_obs[i], color=c, markersize=6)
            scatter!(ax1, [traj_x_obs[i][][end]], [traj_y_obs[i][][end]], 
                    color=c, markersize=15, marker='●', strokewidth=2, strokecolor=:white)
            
            # Right panel
            lines!(ax2, traj_x_obs[i], traj_y_obs[i], color=c, linewidth=2)
            scatter!(ax2, traj_x_obs[i], traj_y_obs[i], color=c, markersize=6)
            scatter!(ax2, [traj_x_obs[i][][end]], [traj_y_obs[i][][end]], 
                    color=c, markersize=15, marker='●', strokewidth=2, strokecolor=:white)
        end
        
        # Add time display
        time_text = Observable(@sprintf("t = %.4f", 0.0))
        Label(fig[0, :], time_text, fontsize=24, tellwidth=false)
        
        # Record video
        record(fig, output_file, 1:n_snapshots; framerate=framerate) do frame_idx
            snap_key = snapshot_keys[frame_idx]
            t = f["snapshots/$snap_key/t"]
            ψ = f["snapshots/$snap_key/psi"]
            
            # Update heatmaps
            phase_obs[] = angle.(ψ)
            density_obs[] = abs2.(ψ)
            
            # Update time display
            time_text[] = @sprintf("t = %.4f", t)
            
            # Update trajectories (if not first frame)
            if frame_idx > 1
                # Get velocity field
                vx_field, vy_field = compute_bohmian_velocity_fft(ψ, kx, ky)
                
                # Update each trajectory
                for i in 1:n_trajectories
                    xp = xps[i]
                    yp = yps[i]
                    
                    # Sample velocity at particle position
                    vx_here = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xp, yp)
                    vy_here = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xp, yp)
                    
                    # Integrate
                    if method == :euler
                        xp += dt_integ * substeps * vx_here
                        yp += dt_integ * substeps * vy_here
                        
                    elseif method == :rk2
                        for _ in 1:substeps
                            k1x, k1y = vx_here, vy_here
                            xmid = xp + 0.5*dt_integ*k1x
                            ymid = yp + 0.5*dt_integ*k1y
                            k2x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xmid, ymid)
                            k2y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xmid, ymid)
                            xp += dt_integ * k2x
                            yp += dt_integ * k2y
                            vx_here, vy_here = k2x, k2y
                        end
                        
                    elseif method == :rk4
                        for _ in 1:substeps
                            k1x, k1y = vx_here, vy_here
                            
                            xmid = xp + 0.5*dt_integ*k1x
                            ymid = yp + 0.5*dt_integ*k1y
                            k2x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xmid, ymid)
                            k2y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xmid, ymid)
                            
                            xmid = xp + 0.5*dt_integ*k2x
                            ymid = yp + 0.5*dt_integ*k2y
                            k3x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xmid, ymid)
                            k3y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xmid, ymid)
                            
                            xend = xp + dt_integ*k3x
                            yend = yp + dt_integ*k3y
                            k4x = bilinear_interpolation(vx_field, x[1], dx, y[1], dy, xend, yend)
                            k4y = bilinear_interpolation(vy_field, x[1], dx, y[1], dy, xend, yend)
                            
                            xp += (dt_integ/6) * (k1x + 2*k2x + 2*k3x + k4x)
                            yp += (dt_integ/6) * (k1y + 2*k2y + 2*k3y + k4y)
                            
                            vx_here = k4x
                            vy_here = k4y
                        end
                    end
                    
                    # Update stored positions
                    xps[i] = xp
                    yps[i] = yp
                    
                    # Add to trajectory arrays
                    push!(traj_xs[i], xp)
                    push!(traj_ys[i], yp)
                    
                    # Update observables
                    traj_x_obs[i][] = copy(traj_xs[i])
                    traj_y_obs[i][] = copy(traj_ys[i])
                end
            end
            
            if frame_idx % 10 == 0
                println("  Rendered frame $frame_idx/$n_snapshots")
            end
        end
        
        println("\n✅ Video saved to: $output_file")
    end
end

# ============================================================================
# Example usage
# ============================================================================

function main()
    # Specify the JLD2 file from your simulation
    jld2_file = "psi_evolution_20251020_123456.jld2"  # Change to your actual file
    
    println("\n" * "="^70)
    println("Creating Bohmian Trajectory Visualization")
    println("="^70)
    
    # ==================================================================
    # Option 1: Single trajectory video
    # ==================================================================
    create_trajectory_video(
        jld2_file,
        -2.0, 2.0,              # Initial position (x0, y0)
        method=:rk4,            # Integration method: :euler, :rk2, :rk4
        substeps=10,            # Substeps between snapshots
        output_file="single_trajectory.mp4",
        framerate=30
    )
    
    # ==================================================================
    # Option 2: Multiple trajectories video (uncomment to use)
    # ==================================================================
    # initial_positions = [
    #     (-2.0, 2.0),
    #     (-1.0, 1.0),
    #     (0.0, 0.0),
    #     (1.0, -1.0),
    #     (2.0, -2.0),
    # ]
    # 
    # create_multiple_trajectory_video(
    #     jld2_file,
    #     initial_positions,
    #     method=:rk4,
    #     substeps=10,
    #     output_file="multiple_trajectories.mp4",
    #     framerate=30
    # )
    
    # ==================================================================
    # Option 3: Export to CSV (if you still need data files)
    # ==================================================================
    # initial_positions = [
    #     (-2.0, 2.0),
    #     (-1.0, 1.0),
    # ]
    # 
    # trajectories = compute_multiple_trajectories(
    #     jld2_file, 
    #     initial_positions,
    #     method=:rk4,
    #     substeps=10
    # )
    # 
    # for (i, traj) in enumerate(trajectories)
    #     x0, y0 = initial_positions[i]
    #     filename = @sprintf("trajectory_%d_x%.2f_y%.2f.csv", i, x0, y0)
    #     CSV.write(filename, traj)
    #     println("\n✅ Saved trajectory $i to $filename")
    # end
    
    println("\n" * "="^70)
    println("All visualizations complete!")
    println("="^70)
end

# Uncomment to run:
# main()

