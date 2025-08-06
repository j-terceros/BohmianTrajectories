using CUDA
using CUDA.CUFFT
using LinearAlgebra           
using AbstractFFTs    
using GLMakie
using Statistics

"""
    The evolution of a 2D entangled quantum state will be carried out
    in the form:
    ψ(x,y,t) = c1 * ψrx(x,t) * ψly(y,t) + c2 * ψlx(x,t) * ψry(y,t)
    where ψrx, ψlx are coherent states in the x direction and ψry, ψly are coherent states in the y direction.
    The 4th-order split-step FFT method of Hatano and Suzuki will be used for the time evolution.

    In this case, quantum decoherence due to continuous measurements will be considered, 
    with the equation having the form:
    iħ ∂ψ/∂t = [-ħ²/(2m) ∇² + V(x,y) + iħ(Wκ + Wγ)]ψ
    where Wκ and Wγ are the decoherence terms.
    Wκ = -κ[ln|ψ|^2 - <ln|ψ|^2>]
    Wγ = γ/2[ln ψ/ψ* - <ln ψ/ψ*>]
    with κ and γ being the continuous measurement resolution and the friction coefficient, respectively.
"""

# ---------------------------

# 1. PARAMETERS AND GRID

# ---------------------------

Nx, Ny = 256, 256
Lx, Ly = 20.0, 20.0
dx, dy = Lx/Nx, Ly/Ny

α0x, α0y = 5/2, 5/2
σx, σy = 0.0, 0.0
ωx, ωy = 1.0, sqrt(3.0)
c2 = 0
c1 = sqrt(1 - c2^2)
tmax, dt = 1.0, 0.0004
tsteps = Int(round(tmax/dt))

# Yoshida 4th order

γ  = (2 + 2^(1/3) + 2^(-1/3)) / 3
dt1 = γ  * dt
dt2 = (1 - 2 * γ) * dt
dt3 = (1 - γ) * dt

# Spatial grids

x = collect(range(-Lx/2, stop=Lx/2 - dx, length=Nx))
y = collect(range(-Ly/2, stop=Ly/2 - dy, length=Ny))

# Frequencies (rad/s)

kx = 2π .* fftfreq(Nx, 1/dx)
ky = 2π .* fftfreq(Ny, 1/dy)

# Replicated fields for GPU

X  = repeat(x, 1, Ny)
Y  = repeat(y', Nx, 1)
KX = repeat(kx, 1, Ny)
KY = repeat(ky', Nx, 1)

# Convert to GPU

KX_gpu = CuArray(KX)
KY_gpu = CuArray(KY)

# Potentials and kinetic operators (CPU)

V_cpu = 0.5 .* (ωx^2 .* X.^2 .+ ωy^2 .* Y.^2)
T_cpu = 0.5 .* (KX.^2 .+ KY.^2)

# Precomputed exponentials on GPU

CUDA.allowscalar(false)  # Disallow scalar indexing

expV1 = CuArray(exp.(-1im .* dt1/2 .* V_cpu))
expV3 = CuArray(exp.(-1im .* dt3/2 .* V_cpu))
expT1 = CuArray(exp.(-1im .* dt1   .* T_cpu))
expT2 = CuArray(exp.(-1im .* dt2   .* T_cpu))

# ---------------------------

# 2. FFT PLANS (GPU)

# ---------------------------

@time begin

# Preallocate buffer for FFT

ψ_k_buf = CuArray{ComplexF64}(undef, Nx, Ny)

# FFT Buffers
dψdx_buf = similar(ψ_k_buf)
dψdy_buf = similar(ψ_k_buf)


# In-place FFT/CUFFT plans
fft_plan  = plan_fft!(ψ_k_buf)
ifft_plan = plan_ifft!(ψ_k_buf)

# ---------------------------

# 3. DEFINITION OF ENTANGLEMENT

# ---------------------------

function coherent1D(x::AbstractVector{<:Real}, α0::Real, σ::Real, ω::Real, t::Real)
    αt = α0 * exp(-1im*(ω*t - σ))
    x̄ = sqrt(2/ω)*real(αt)
    p̄ = sqrt(2*ω)*imag(αt)
    Δx = sqrt(1/(2ω))
    θ = -ω*t/2 + (abs(α0)^2*sin(2*ω*t - 2σ))/2
    φ = exp(1im*θ)
    pref = (ω/π)^(1/4)
    return pref .* φ .* exp.(-((x.-x̄).^2)/(2*Δx)^2) .* exp.(1im*p̄.*x)
end

function entangled_ψ(c1, c2, t)
    ψrx = coherent1D(x, α0x, σx, ωx, t)
    ψlx = coherent1D(x, α0x, σx+π, ωx, t)
    ψry = coherent1D(y, α0y, σy, ωy, t)
    ψly = coherent1D(y, α0y, σy+π, ωy, t)
    return c1 .* ψrx .* reshape(ψly,1,:) .+ c2 .* ψlx .* reshape(ψry,1,:)
end

# ---------------------------

# 4. DECOHERENCE OPERATORS

#----------------------------

function decoherence_terms(ψ, dt1, dt3; κ = -1.0, γ = 1.0)
    ln_ψ2 = log.(abs2.(ψ))
    ln_ψψ_conj = log.(ψ ./ conj(ψ))

    Wκ = -κ * (ln_ψ2 .- mean(ln_ψ2))
    Wγ = γ/2 * (ln_ψψ_conj .- mean(ln_ψψ_conj))

    # im due to the multiplication of -i for the SSFM
    im_H_dec = Wκ + Wγ  # ħ = 1 
    expH_dec1 = CuArray(exp.(dt1/2 .* im_H_dec))
    expH_dec3 = CuArray(exp.(dt3/2 .* im_H_dec))

    return expH_dec1, expH_dec3
end

# ---------------------------

# 5. EVOLUTION FUNCTION

# ---------------------------

function run_splitstep_bohmian()
    # CPU initial state
    ψ0_cpu = Array{ComplexF64}(undef, Nx, Ny)
    ψ0_cpu .= entangled_ψ(c1, c2, 0.0)
    # Convert to GPU
    ψ = CuArray(ψ0_cpu)

    @info("Initializing evolution in GPU...")
    for n in 1:tsteps
        # Compute decoherence terms
        expH_dec1, expH_dec3 = decoherence_terms(ψ, dt1, dt3)

        # Split‐step FFT 4º order
        ψ .*= expV1 .* expH_dec1
        copyto!(ψ_k_buf, ψ)
        mul!(ψ_k_buf, fft_plan, ψ_k_buf)
        ψ_k_buf .*= expT1
        mul!(ψ_k_buf, ifft_plan, ψ_k_buf)
        copyto!(ψ, ψ_k_buf)

        ψ .*= expV3 .* expH_dec3
        copyto!(ψ_k_buf, ψ)
        mul!(ψ_k_buf, fft_plan, ψ_k_buf)
        ψ_k_buf .*= expT2
        mul!(ψ_k_buf, ifft_plan, ψ_k_buf)
        copyto!(ψ, ψ_k_buf)

        ψ .*= expV3 .* expH_dec3
        copyto!(ψ_k_buf, ψ)
        mul!(ψ_k_buf, fft_plan, ψ_k_buf)
        ψ_k_buf .*= expT1
        mul!(ψ_k_buf, ifft_plan, ψ_k_buf)
        copyto!(ψ, ψ_k_buf)

        ψ .*= expV1 .* expH_dec1

    end
    @info("Evolution completed.")
    return Array(ψ)
end

# ---------------------------

# 6. ANIMATION

# ---------------------------

function animate_psi_gpu(
    x, y,                     # grid and initial state
    fft_plan, ifft_plan,     # buffers and preallocated FFT plans
    expV1, expT2, expV3,     # Hatano-Suzuki exponents
    tsteps;
    filename::String="psi_evolution_gpu.mp4",
    framerate::Int=24,
    maxframes::Int=800
)
    @info "🟢 Preparing animation GPU..."
    # 1) Initial state in CPU
    ψ0_cpu = Array{ComplexF64}(undef, Nx, Ny)
    ψ0_cpu .= entangled_ψ(c1, c2, 0.0)
    ψ = CuArray(ψ0_cpu)

    # 2) Figure and initial surface
    fig = Figure(size=(800, 400))
    ax  = Axis3(fig[1,1], title="Evolution |ψ|² [GPU]",
                xlabel="x", ylabel="y", zlabel="|ψ|²")
    surf = surface!(ax, x, y, abs2.(ψ0_cpu), colormap=:viridis)



    # 3) Frame indices
    step_idx = max(1, floor(Int, tsteps ÷ maxframes))
    idxs     = 1:step_idx:tsteps
    @info "🖼️ Recording $(length(idxs)) frames a ‘($filename)’"

    # 4) Animation recording
    GLMakie.record(fig, filename, idxs; framerate=framerate) do frame
        for _ in 1:step_idx
            expH_dec1, expH_dec3 = decoherence_terms(ψ, dt1, dt3)

            ψ .*= expV1 .* expH_dec1
            copyto!(ψ_k_buf, ψ)
            mul!(ψ_k_buf, fft_plan, ψ_k_buf)
            ψ_k_buf .*= expT1
            mul!(ψ_k_buf, ifft_plan, ψ_k_buf)
            copyto!(ψ, ψ_k_buf)

            ψ .*= expV3 .* expH_dec3
            copyto!(ψ_k_buf, ψ)
            mul!(ψ_k_buf, fft_plan, ψ_k_buf)
            ψ_k_buf .*= expT2
            mul!(ψ_k_buf, ifft_plan, ψ_k_buf)
            copyto!(ψ, ψ_k_buf)

            ψ .*= expV3 .* expH_dec3
            copyto!(ψ_k_buf, ψ)
            mul!(ψ_k_buf, fft_plan, ψ_k_buf)
            ψ_k_buf .*= expT1
            mul!(ψ_k_buf, ifft_plan, ψ_k_buf)
            copyto!(ψ, ψ_k_buf)

            ψ .*= expV1 .* expH_dec1
        end

        # 5) Update the surface with the current |ψ|²
        surf[3] = abs2.(Array(ψ))
    end

    @info "✅ Animation GPU saved as ‘$(filename)’."
end

animate_psi_gpu(
    x, y,
    fft_plan, ifft_plan,
    expV1, expT2, expV3,
    tsteps;
    filename="decoh_animation.mp4",
    framerate=24,
    maxframes=800
)

end