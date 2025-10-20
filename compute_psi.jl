using LinearAlgebra
using CUDA
using CUDA.CUFFT
using FFTW
using JLD2
using Printf
using ProgressBars
using Dates

# -------------------------
# Constantes y parámetros
# -------------------------

function main_psi(TT::Type{<:AbstractFloat} = Float32)
    # Machine precision for the chosen type
    ε = eps(TT)
    ħ = TT(1)
    m = TT(1)
    κ = TT(0)
    ν = TT(0)
    
    println("\n" * "="^70)
    println("Starting Quantum Evolution Simulation")
    println("Precision: $TT (ε = $ε)")
    println("="^70)
    
    Nx, Ny = 512, 512 
    Lx, Ly = TT(20.0), TT(20.0)
    dx, dy = Lx / Nx, Ly / Ny
    α0x, α0y = TT(5/2), TT(5/2)
    σx, σy = TT(0.0), TT(0.0)
    ωx, ωy = TT(2.0), TT(sqrt(3.0))
    dt = TT(0.0001)
    tmax = TT(7.0)
    Nt = Int(cld(tmax, dt))

    # Unique filename per run
    run_id = @sprintf "k=%3.1f_n=%3.1f_c2=bell" κ ν
    out_jld2 = "psi_evolution_$(run_id).jld2"

    # Save ψ snapshot every 'save_every' timesteps
    save_every = 10

    c2 = TT(sqrt(2)/2)
    c1 = TT(sqrt(1 - c2^2))

    # -------------------------
    # Mallas CPU
    # -------------------------
    x  = TT.(collect(range(-Lx/2, stop=Lx/2, length=Nx+1))[1:end-1])
    y  = TT.(collect(range(-Ly/2, stop=Ly/2, length=Ny+1))[1:end-1])
    kx = TT.(fftfreq(Nx, 2π/dx))
    ky = TT.(fftfreq(Ny, 2π/dy))
    KX = repeat(kx, 1, Ny)
    KY = repeat(ky', Nx, 1)
    X  = repeat(x,  1, Ny)
    Y  = repeat(y', Nx, 1)

    # Suzuki–Yoshida (como tu loop estable)
    γ  = TT(1/(4 - cbrt(4)))
    dt1 = 0.5 * γ  * dt
    dt2 = γ  * dt
    dt3 = 0.5 * (1 - 3 * γ) * dt
    dt4 = (1 - 4 * γ) * dt

    V = TT.(0.5 * (ωx^2 .* X.^2 .+ ωy^2 .* Y.^2))
    T = TT.(0.5 * (KX.^2 .+ KY.^2))

    # Transfer operators to GPU with explicit type
    CT = Complex{TT}
    expV2  = CuArray{CT}(exp.(-1im * dt2 * V))
    expV6  = CuArray{CT}(exp.(-1im * dt4 * V))
    expT1  = CuArray{CT}(exp.(-1im * dt1 * T))
    expT3  = CuArray{CT}(exp.(-1im * dt2 * T))
    expT5  = CuArray{CT}(exp.(-1im * dt3 * T))

    # -------------------------
    # Auxiliares
    # -------------------------
    function decoherence_mul!(ψ, dt_loc)
        ρ = abs2.(ψ)
        Z = sum(ρ) + ε
        w = ρ ./ Z
        lnρ = log.(ρ .+ ε)
        mean_lnρ = sum(lnρ .* w)
        Wκ = -κ .* (lnρ .- mean_lnρ)
        φ = angle.(ψ)
        S = sum(w .* sin.(φ));  C = sum(w .* cos.(φ))
        φ̄ = (abs(S) + abs(C) < ε) ? 0 : atan(S, C)
        δφ = rem2pi.(φ .- φ̄, RoundNearest)
        @. ψ = ψ * exp(dt_loc * Wκ)
        @. ψ = ψ * cis(-ν * dt_loc * δφ)
        return nothing
    end

    function coherent1D(x, α0, σ, ω, t)
        αt = α0 * exp(-1im * (ω * t - σ))
        x̄ = sqrt(2/ω) * real(αt)
        p̄ = sqrt(2*ω) * imag(αt)
        Δx = sqrt(1/(2*ω))
        θ_full = -ω*t/2 + (abs(α0)^2 * sin(2*ω*t - 2*σ))/2
        φ = exp(1im * θ_full)
        gauss = @. exp(-(x - x̄)^2 / (2*Δx)^2)
        plane = @. exp(1im * p̄ * x)
        pref = (ω/π)^(1/4)
        return pref * φ .* gauss .* plane
    end

    function entangled_ψ(c1, c2, t)
        ψrx = coherent1D(x, α0x, σx,   ωx, t)
        ψlx = coherent1D(x, α0x, σx+π, ωx, t)
        ψry = coherent1D(y, α0y, σy,   ωy, t)
        ψly = coherent1D(y, α0y, σy+π, ωy, t)
        return c1 .* (ψrx .* reshape(ψly, 1, :)) .+ c2 .* (ψlx .* reshape(ψry, 1, :))
    end

    # -------------------------
    # Inicialización
    # -------------------------
    CT = Complex{TT}
    ψ_sim = CuArray{CT}(entangled_ψ(c1, c2, TT(0.0)))

    # -------------------------
    # Simulación + guardado
    # -------------------------
    jldopen(out_jld2, "w"; compress=true) do f
        # Save metadata for post-processing
        f["meta/x"] = x;  f["meta/y"] = y
        f["meta/Nx"] = Nx; f["meta/Ny"] = Ny
        f["meta/Lx"] = Lx; f["meta/Ly"] = Ly
        f["meta/dx"] = dx; f["meta/dy"] = dy
        f["meta/dt"] = dt; f["meta/save_every"] = save_every
        f["meta/c1"] = c1; f["meta/c2"] = c2
        f["meta/omegax"] = ωx; f["meta/omegay"] = ωy

        # Function to save ψ snapshot
        function save_snapshot!(f, k, t, ψ_gpu)
            gname = @sprintf("snapshots/it_%06d", k)
            f["$gname/t"]   = t
            f["$gname/psi"] = Array(ψ_gpu)   # ComplexF64 Nx×Ny
        end

        # Save initial snapshot (k=0)
        save_snapshot!(f, 0, 0.0, ψ_sim)

        # Main time evolution loop
        for k in ProgressBar(1:Nt)
            # Suzuki-Yoshida 4th order symplectic integrator
            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT1
            CUDA.CUFFT.ifft!(ψ_sim)

            decoherence_mul!(ψ_sim, dt2)
            ψ_sim .*= expV2

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT3
            CUDA.CUFFT.ifft!(ψ_sim)

            decoherence_mul!(ψ_sim, dt2)
            ψ_sim .*= expV2

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT5
            CUDA.CUFFT.ifft!(ψ_sim)

            decoherence_mul!(ψ_sim, dt4)
            ψ_sim .*= expV6

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT5
            CUDA.CUFFT.ifft!(ψ_sim)

            decoherence_mul!(ψ_sim, dt2)
            ψ_sim .*= expV2

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT3
            CUDA.CUFFT.ifft!(ψ_sim)

            decoherence_mul!(ψ_sim, dt2)
            ψ_sim .*= expV2

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT1
            CUDA.CUFFT.ifft!(ψ_sim)

            # Save snapshot at regular intervals
            if (k % save_every) == 0
                t = k * dt
                save_snapshot!(f, k, t, ψ_sim)
            end
        end
    end  # Close JLD2 file

    # Calculate number of saved snapshots
    nsaved = Int(floor(Nt / save_every)) + 1  # +1 for initial snapshot
    
    println("\n" * "="^70)
    println("✅ Simulation Complete!")
    println("="^70)
    println("   Precision: $TT")
    println("   Run ID: $run_id")
    println("   Output file: $out_jld2")
    println("   Snapshots saved: $nsaved")
    println("   Time step (dt): $dt")
    println("   Save interval: every $save_every steps ($(save_every*dt) time units)")
    println("   Total simulation time: $(Nt*dt)")
    println("\n   Use compute_trajectories.jl to visualize Bohmian trajectories.")
    println("="^70 * "\n")
end
