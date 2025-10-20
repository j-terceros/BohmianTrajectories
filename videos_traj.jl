using LinearAlgebra
using CUDA
using CUDA.CUFFT
using FFTW
using JLD2
using CSV, DataFrames
using Printf
using ProgressBars
using Dates

# -------------------------
# Constantes y parámetros
# -------------------------
const ħ = 1.0
const m = 1.0
const κ = 0.0
const ν = 0.0
const ε = 1e-14

function main()
    Nx, Ny = 1024, 1024
    Lx, Ly = 20.0, 20.0
    dx, dy = Lx / Nx, Ly / Ny
    α0x, α0y = 5/2, 5/2
    σx, σy = 0.0, 0.0
    ωx, ωy = 1.0, sqrt(3.0)
    dt = 0.00001
    tmax = 100.0
    Nt = Int(cld(tmax, dt))

    # ---------- Opción B: nombres únicos por corrida ----------
    run_id  = Dates.format(now(), "yyyymmdd_HHMMSS")
    out_jld2 = "run_snapshots_$(run_id).jld2"
    out_csv  = "trajectory_samples_$(run_id).csv"

    # cada 'save_every' pasos se guarda snapshot completo (psi, vx, vy)
    save_every = 40

    c2 = 2e-5
    c1 = sqrt(1 - c2^2)

    # -------------------------
    # Mallas CPU
    # -------------------------
    x  = collect(range(-Lx/2, stop=Lx/2, length=Nx+1))[1:end-1]
    y  = collect(range(-Ly/2, stop=Ly/2, length=Ny+1))[1:end-1]
    kx = fftfreq(Nx, 2π/dx)
    ky = fftfreq(Ny, 2π/dy)
    KX = repeat(kx, 1, Ny)
    KY = repeat(ky', Nx, 1)
    X  = repeat(x,  1, Ny)
    Y  = repeat(y', Nx, 1)

    # Suzuki–Yoshida (como tu loop estable)
    γ  = 1/(4 - cbrt(4))
    dt1 = 0.5 * γ  * dt
    dt2 = γ  * dt
    dt3 = 0.5 * (1 - 3 * γ) * dt
    dt4 = (1 - 4 * γ) * dt

    V = 0.5 * (ωx^2 .* X.^2 .+ ωy^2 .* Y.^2)
    T = 0.5 * (KX.^2 .+ KY.^2)

    # A GPU
    KX_gpu = CuArray(KX)
    KY_gpu = CuArray(KY)
    expV2  = CuArray(exp.(-1im * dt2 * V))
    expV6  = CuArray(exp.(-1im * dt4 * V))
    expT1  = CuArray(exp.(-1im * dt1 * T))
    expT3  = CuArray(exp.(-1im * dt2 * T))
    expT5  = CuArray(exp.(-1im * dt3 * T))

    # -------------------------
    # Auxiliares
    # -------------------------
    function decoherence_mul!(ψ::CuArray{ComplexF64}, dt_loc)
        ρ = abs2.(ψ)
        Z = sum(ρ) + ε
        w = ρ ./ Z
        lnρ = log.(ρ .+ ε)
        mean_lnρ = sum(lnρ .* w)
        Wκ = -κ .* (lnρ .- mean_lnρ)
        φ = angle.(ψ)
        S = sum(w .* sin.(φ));  C = sum(w .* cos.(φ))
        φ̄ = (abs(S) + abs(C) < 1e-14) ? 0.0 : atan(S, C)
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

    # Bohm (derivadas espectrales en GPU)
    function bohmian_velocity_fft!(
            ψ::CuArray{ComplexF64}, KXg::CuArray{Float64}, KYg::CuArray{Float64},
            ψ_hat::CuArray{ComplexF64}, dψdx_hat::CuArray{ComplexF64}, dψdy_hat::CuArray{ComplexF64},
            dψdx::CuArray{ComplexF64}, dψdy::CuArray{ComplexF64},
            v_x::CuArray{Float64}, v_y::CuArray{Float64}
        )
        copyto!(ψ_hat, ψ)
        CUDA.CUFFT.fft!(ψ_hat)
        @. dψdx_hat = (1im) * KXg * ψ_hat
        @. dψdy_hat = (1im) * KYg * ψ_hat
        copyto!(dψdx, dψdx_hat); copyto!(dψdy, dψdy_hat)
        CUDA.CUFFT.ifft!(dψdx);   CUDA.CUFFT.ifft!(dψdy)
        @. v_x = imag( dψdx / (ψ + 1e-15) )
        @. v_y = imag( dψdy / (ψ + 1e-15) )
        return nothing
    end

    # Muestreo bilineal (GPU) de velocidad en (xp, yp)
    function sample_velocity_kernel!(
        vx::CuDeviceArray{Float64,2}, vy::CuDeviceArray{Float64,2},
        x0::Float64, dx::Float64, y0::Float64, dy::Float64,
        xp::Float64, yp::Float64,
        out_vx::CuDeviceVector{Float64}, out_vy::CuDeviceVector{Float64}
    )
        i = (xp - x0) / dx;  j = (yp - y0) / dy
        ix = Int(floor(i)) + 1; jx = Int(floor(j)) + 1
        wx = i - floor(i);      wy = j - floor(j)
        ix1 = clamp(ix, 1, size(vx,1)-1); jx1 = clamp(jx, 1, size(vx,2)-1)
        v00x = vx[ix1, jx1];   v10x = vx[ix1+1, jx1]
        v01x = vx[ix1, jx1+1]; v11x = vx[ix1+1, jx1+1]
        v00y = vy[ix1, jx1];   v10y = vy[ix1+1, jx1]
        v01y = vy[ix1, jx1+1]; v11y = vy[ix1+1, jx1+1]
        sx = (1-wx)(1-wy)*v00x + wx(1-wy)*v10x + (1-wx)*wy*v01x + wx*wy*v11x
        sy = (1-wx)(1-wy)*v00y + wx(1-wy)*v10y + (1-wx)*wy*v01y + wx*wy*v11y
        out_vx[1] = sx; out_vy[1] = sy
        return
    end
    function sample_velocity_gpu!(vx::CuArray{Float64,2}, vy::CuArray{Float64,2},
                                  x0::Float64, dx::Float64, y0::Float64, dy::Float64,
                                  xp::Float64, yp::Float64,
                                  out_vx_gpu::CuArray{Float64,1},
                                  out_vy_gpu::CuArray{Float64,1})
        @cuda threads=1 sample_velocity_kernel!(vx, vy, x0, dx, y0, dy, xp, yp, out_vx_gpu, out_vy_gpu)
    end

    # -------------------------
    # Inicialización
    # -------------------------
    ψ_sim = CuArray(complex.(entangled_ψ(c1, c2, 0.0)))
    ψ_hat    = similar(ψ_sim)
    dψdx_hat = similar(ψ_sim)
    dψdy_hat = similar(ψ_sim)
    dψdx     = similar(ψ_sim)
    dψdy     = similar(ψ_sim)
    v_x = CuArray(zeros(Float64, size(ψ_sim)))
    v_y = CuArray(zeros(Float64, size(ψ_sim)))
    out_vx_gpu = CuArray(zeros(Float64, 1))
    out_vy_gpu = CuArray(zeros(Float64, 1))

    # Partícula (Refs para scope seguro)
    x0p, y0p = -2.0, 2.0
    xp = Ref(x0p)
    yp = Ref(y0p)

    # CSV buffers (solo en pasos guardados) + índice como Ref
    nsave = Int(floor(Nt / save_every)) + 1
    times = Vector{Float64}(undef, nsave)
    posx  = Vector{Float64}(undef, nsave)
    posy  = Vector{Float64}(undef, nsave)
    velx  = Vector{Float64}(undef, nsave)
    vely  = Vector{Float64}(undef, nsave)
    save_idx = Ref(1)

    # -------------------------
    # Simulación + guardado
    # -------------------------
    jldopen(out_jld2, "w") do f
        # meta
        f["meta/x"] = x;  f["meta/y"] = y
        f["meta/Nx"] = Nx; f["meta/Ny"] = Ny
        f["meta/dx"] = dx; f["meta/dy"] = dy
        f["meta/dt"] = dt; f["meta/save_every"] = save_every

        # función para snapshot
        function save_snapshot!(f, k::Int, t::Float64, ψ_gpu, vx_gpu, vy_gpu, xpp::Float64, ypp::Float64)
            gname = @sprintf("snapshots/it_%06d", k)
            f["$gname/t"]  = t
            f["$gname/xp"] = xpp
            f["$gname/yp"] = ypp
            f["$gname/psi"] = Array(ψ_gpu)   # ComplexF64 Nx×Ny
            f["$gname/vx"]  = Array(vx_gpu)  # Float64  Nx×Ny
            f["$gname/vy"]  = Array(vy_gpu)  # Float64  Nx×Ny
        end

        # snapshot inicial (k=0)
        k = 0
        t = 0.0
        bohmian_velocity_fft!(ψ_sim, KX_gpu, KY_gpu, ψ_hat, dψdx_hat, dψdy_hat, dψdx, dψdy, v_x, v_y)
        save_snapshot!(f, k, t, ψ_sim, v_x, v_y, xp[], yp[])

        # CSV inicial
        sample_velocity_gpu!(v_x, v_y, x[1], dx, y[1], dy, xp[], yp[], out_vx_gpu, out_vy_gpu)
        CUDA.synchronize()
        times[save_idx[]] = t
        posx[save_idx[]]  = xp[]
        posy[save_idx[]]  = yp[]
        velx[save_idx[]]  = Array(out_vx_gpu)[1]
        vely[save_idx[]]  = Array(out_vy_gpu)[1]

        # bucle temporal
        for k in ProgressBar(1:Nt)
            # Paso temporal (idéntico al tuyo)
            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT1
            CUDA.CUFFT.ifft!(ψ_sim)

            #decoherence_mul!(ψ_sim, dt2)
            ψ_sim .*= expV2

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT3
            CUDA.CUFFT.ifft!(ψ_sim)

            #decoherence_mul!(ψ_sim, dt2)
            ψ_sim .*= expV2

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT5
            CUDA.CUFFT.ifft!(ψ_sim)

            #decoherence_mul!(ψ_sim, dt4)
            ψ_sim .*= expV6

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT5
            CUDA.CUFFT.ifft!(ψ_sim)

            #decoherence_mul!(ψ_sim, dt2)
            ψ_sim .*= expV2

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT3
            CUDA.CUFFT.ifft!(ψ_sim)

            #decoherence_mul!(ψ_sim, dt2)
            ψ_sim .*= expV2

            CUDA.CUFFT.fft!(ψ_sim)
            @. ψ_sim = ψ_sim * expT1
            CUDA.CUFFT.ifft!(ψ_sim)

            # Campo de velocidades (GPU)
            bohmian_velocity_fft!(ψ_sim, KX_gpu, KY_gpu, ψ_hat, dψdx_hat, dψdy_hat, dψdx, dψdy, v_x, v_y)

            # Velocidad en posición de la partícula
            sample_velocity_gpu!(v_x, v_y, x[1], dx, y[1], dy, xp[], yp[], out_vx_gpu, out_vy_gpu)
            CUDA.synchronize()
            vx_here = Array(out_vx_gpu)[1]
            vy_here = Array(out_vy_gpu)[1]

            # Euler con dt
            xp[] += dt * vx_here
            yp[] += dt * vy_here

            # Guardado con stride
            if (k % save_every) == 0
                t = k * dt
                save_snapshot!(f, k, t, ψ_sim, v_x, v_y, xp[], yp[])

                save_idx[] += 1
                times[save_idx[]] = t
                posx[save_idx[]]  = xp[]
                posy[save_idx[]]  = yp[]
                velx[save_idx[]]  = vx_here
                vely[save_idx[]]  = vy_here
            end

            # (opcional) envolver a dominio periódico
            # xp[] = mod(xp[] - x[1], x[end]-x[1]) + x[1]
            # yp[] = mod(yp[] - y[1], y[end]-y[1]) + y[1]
        end
    end  # cierra JLD2

    # recorta CSV buffers según último índice guardado
    last = save_idx[]
    times = times[1:last]
    posx  = posx[1:last]
    posy  = posy[1:last]
    velx  = velx[1:last]
    vely  = vely[1:last]

    df = DataFrame(time = times, x = posx, y = posy, vx = velx, vy = vely)
    CSV.write(out_csv, df)

    println("✅ Run ID: $run_id")
    println("✅ Snapshots COMPLETOS (psi, vx, vy) en $out_jld2")
    println("✅ Trayectoria muestreada (time,x,y,vx,vy) en $out_csv")
    println("Guardados: $last filas, stride = $(save_every) (≈ $(save_every*dt) s), dt = $dt")
end
