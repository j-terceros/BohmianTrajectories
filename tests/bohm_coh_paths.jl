####################################################################
#   Bohmian Trajectory 2D for a Particle: SSFM + Interpolation     #
#####################################################################

# Required Libraries
using GLMakie            # For visualization
using FFTW               # For Fast Fourier Transforms
using LinearAlgebra      # For Linear Algebra operations
using Interpolations     # For interpolation techniques

let
    # ===== 1. SIMULATION PARAMETERS =====
    Nx, Ny = 256, 256           # Resolution of the grid (
    Lx, Ly = 20.0, 20.0        # Spatial domain sizes
    dx, dy = Lx / Nx, Ly / Ny  # Grid spacings
    a0x, a0y = 1.0 + 0im, 1.0 + 0im  # Initial coherent states in the x and y directions
    ωx, ωy = 6.0, 7.0          # Frequencies in the x and y directions
    tmax, dt = 100.0, 0.01     # Max time and time step
    tsteps = Int(round(tmax / dt))  # Number of time steps

    # ===== 2. GRID AND FFT OPERATORS =====
    # Define 1D spatial grids for x and y
    x = collect(range(-Lx / 2, stop=Lx / 2, length=Nx + 1))[1:end - 1]  # Nx points in x
    y = collect(range(-Ly / 2, stop=Ly / 2, length=Ny + 1))[1:end - 1]  # Ny points in y

    # FFTW convention: fftfreq(N, fs)
    kx = fftfreq(Nx, 2π / dx)      # Fourier frequencies in the x direction
    ky = fftfreq(Ny, 2π / dy)      # Fourier frequencies in the y direction
    KX = repeat(kx, 1, Ny)         # (Nx, Ny) matrix for KX
    KY = repeat(ky', Nx, 1)        # (Nx, Ny) matrix for KY

    X = repeat(x, 1, Ny)           # (Nx, Ny) matrix for x coordinates
    Y = repeat(y', Nx, 1)          # (Nx, Ny) matrix for y coordinates

    # ===== 3. SPLIT-STEP OPERATORS =====
    # Potential V for harmonic oscillator (2D)
    V = 0.5 * (ωx^2 * X.^2 .+ ωy^2 * Y.^2)
    expV = exp.(-1im * dt * V)     # Potential exponential term for split-step

    # Kinetic energy operator (Fourier space)
    T = 0.5 * (KX.^2 .+ KY.^2)
    expT = exp.(-1im * dt / 2 * T) # Kinetic energy exponential term for split-step

    # ===== 4. COHERENT STATES (1D and 2D) =====
    # Function for coherent 1D Gaussian states
    function coherent1D(x::AbstractVector{<:Real}, α0::ComplexF64, ω::Real, t::Real)
        σ = angle(α0)                      # Phase of initial condition
        αt = α0 * exp(-1im * ω * t)         # Time evolution of coherent state
        x̄ = sqrt(2 / ω) * real(αt)         # Mean position of the wave packet
        p̄ = sqrt(2 * ω) * imag(αt)         # Mean momentum of the wave packet
        Δx = sqrt(1 / (2 * ω))              # Width of the wave packet
        θ_full = -ω * t / 2 + (abs(α0)^2 * sin(2 * ω * t - 2 * σ)) / 2
        φ = exp(1im * θ_full)               # Time-dependent phase factor
        gauss = @. exp(- (x - x̄)^2 / (2 * Δx)^2)  # Gaussian envelope
        plane = @. exp(1im * p̄ * x)         # Plane wave factor
        pref = (ω / π)^(1/4)                 # Normalization constant
        return pref * φ .* gauss .* plane   # Coherent state in position space
    end

    # Function for 2D coherent state by taking the outer product of two 1D states
    function coherent2D(x, y, α0x, ωx, α0y, ωy, t)
        ψx = coherent1D(x, α0x, ωx, t)      # 1D coherent state in x-direction
        ψy = coherent1D(y, α0y, ωy, t)      # 1D coherent state in y-direction
        ψ = ψx .* reshape(ψy, 1, :)         # Outer product (Nx, Ny)
        return ComplexF64.(ψ)                # Complex-valued wave function
    end

    # ===== 5. BOHMIAN VELOCITY FUNCTIONS (FFT-based, robust) =====
    # Compute Bohmian velocity in 2D using FFT
    function bohmian_velocity_fft(ψ::AbstractMatrix{<:ComplexF64}, KX, KY)
        # ψ: Wave function (Nx, Ny), KX, KY: Fourier space wave numbers (Nx, Ny)
        ψ̂ = fft(ψ)                           # Fourier transform of ψ
        dψdx_hat = 1im * KX .* ψ̂            # Fourier transform of ∂ψ/∂x
        dψdy_hat = 1im * KY .* ψ̂            # Fourier transform of ∂ψ/∂y
        dψdx = ifft(dψdx_hat)                # Inverse FFT to get ∂ψ/∂x in real space
        dψdy = ifft(dψdy_hat)                # Inverse FFT to get ∂ψ/∂y in real space
        v_x = imag.(dψdx ./ ψ)               # Bohmian velocity in x (imaginary part of derivative)
        v_y = imag.(dψdy ./ ψ)               # Bohmian velocity in y (imaginary part of derivative)
        return v_x, v_y                      # Return both components (Nx, Ny)
    end

    # ===== 6. INITIALIZATION OF ψ, BUFFERS, AND TRAJECTORY =====
    # Initialize the wave function ψ at t=0 (Nx, Ny)
    ψ = coherent2D(x, y, a0x, ωx, a0y, ωy, 0.0)
    ψ_k = similar(ψ)                       # Buffer for FFT

    # FFT and IFFT plans for optimization
    fft_plan = plan_fft(ψ, flags=FFTW.MEASURE)
    ifft_plan = plan_ifft(ψ, flags=FFTW.MEASURE)

    # Initial probability density |ψ|²
    ρ0 = abs2.(ψ)                          # (Nx, Ny)

    # Find the maximum of |ψ|² to get the initial Bohmian position
    indmax = argmax(ρ0)                     # Index of the maximum
    i0, j0 = Tuple(CartesianIndices(ρ0)[indmax])  # Convert to 2D indices
    x0 = x[i0]                             # Initial x position
    y0 = y[j0]                             # Initial y position

    # Initialize Bohmian trajectory arrays
    trajectory_x = zeros(Float64, tsteps + 1)  # (tsteps + 1) to store trajectory
    trajectory_y = zeros(Float64, tsteps + 1)
    trajectory_x[1] = x0                       # Starting x position
    trajectory_y[1] = y0                       # Starting y position

    # ===== 7. TRAJECTORY INTEGRATION =====
    xp, yp = x0, y0  # Initial particle position

    # Time-stepping loop
    for step in 1:tsteps
        # ----- Evolve ψ with Split-Step Fourier Method (SSFM) -----
        mul!(ψ_k, fft_plan, ψ)         # Apply FFT to ψ
        ψ_k .*= expT                    # Apply kinetic operator
        mul!(ψ, ifft_plan, ψ_k)         # Inverse FFT to obtain updated ψ
        ψ .*= expV                       # Apply potential operator
        mul!(ψ_k, fft_plan, ψ)         # Apply FFT to updated ψ
        ψ_k .*= expT                    # Apply kinetic operator again
        mul!(ψ, ifft_plan, ψ_k)         # Final inverse FFT to get updated ψ

        # ----- Compute Bohmian velocities -----
        v_x, v_y = bohmian_velocity_fft(ψ, KX, KY)  # Compute velocities in x and y
        # Bilinear interpolation to estimate velocities at current position
        itp_vx = interpolate((x, y), v_x, Gridded(Linear()))
        itp_vy = interpolate((x, y), v_y, Gridded(Linear()))
        # Evaluate velocity at current position (xp, yp)
        vx_here = itp_vx(xp, yp)
        vy_here = itp_vy(xp, yp)

        # Euler step for position update
        xp_new = xp + dt * vx_here
        yp_new = yp + dt * vy_here

        # Store the new position
        trajectory_x[step + 1] = xp_new
        trajectory_y[step + 1] = yp_new

        # Update position for next iteration
        xp, yp = xp_new, yp_new

        # Control: if the particle leaves the domain, stop the integration
        if xp < minimum(x) || xp > maximum(x) || yp < minimum(y) || yp > maximum(y)
            trajectory_x = trajectory_x[1:step + 1]
            trajectory_y = trajectory_y[1:step + 1]
            @warn "Particle left the domain at step $step!"
            break
        end
    end

    # ===== 8. VISUALIZATION OF THE TRAJECTORY =====
    # Visualize the final results using Makie
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Bohmian Trajectory + Final |ψ|²")
    hm = heatmap!(ax, x, y, abs2.(ψ)', colormap=:viridis)  # Final probability density (|ψ|²) as a heatmap
    lines!(ax, trajectory_x, trajectory_y, color=:red, linewidth=2)  # Plot trajectory in red
    scatter!(ax, [trajectory_x[1]], [trajectory_y[1]], color=:green, markersize=12, label="Start")  # Starting point
    scatter!(ax, [trajectory_x[end]], [trajectory_y[end]], color=:orange, markersize=10, label="End")  # End point
    axislegend(ax)  # Add a legend to the plot
    display(fig)  # Display the figure
end