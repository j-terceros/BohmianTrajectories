##############################
# Schrödinger + Decoherencia #
# SplitODE + GPU (1D)        #
# Precisión: Float64         #
##############################

# -------------------- IMPORTS --------------------
using CUDA                    # CuArray, GPU
using CUDA.CUFFT              # planes cuFFT
using LinearAlgebra           # norm, etc.
using DifferentialEquations   # SplitODEProblem + solve
using OrdinaryDiffEq          # KenCarp58 (IMEX)
using LinearSolve             # KrylovJL_GMRES (JFNK)
using FFTW                    # fftfreq (k)
# using GLMakie               # plot (opcional)

CUDA.allowscalar(false)       # evita indexado escalar en GPU

# -------------------- CONFIG --------------------
Base.@kwdef struct Config
    dims::Int                         = 1            # SOLO 1D
    Nx::Int                           = 512
    Lx::Float64                       = 30.0
    tspan::Tuple{Float64,Float64}     = (0.0, 1.0)
    ħ::Float64                        = 1.0
    m::Float64                        = 1.0
    κ::Float64                        = -1.0
    ν::Float64                        = 1.0
    ωx::Float64                       = 1.0
    small::Float64                    = 1e-12        # regularización más robusta
    reltol::Float64                   = 1e-9
    abstol::Float64                   = 1e-9
    maxiters::Int                     = 1_000_000    # evita lazos eternos
    saveat::Float64                   = 0.01
    dt::Union{Nothing,Float64}        = nothing
    dtmax::Union{Nothing,Float64}     = nothing      # puedes activarlo tras estabilizar
    # unwrap / nodos
    rho_floor::Float64                = 1e-12        # máscara en nodos
    unwrap_kc::Float64                = Inf          # low-pass del unwrap (Inf = off)
    # estado coherente inicial
    α0x::ComplexF64                   = 2.5 + 0.0im
    σx::Float64                       = 0.0
end

Base.@kwdef struct PhysParams{T}
    ħ::T = one(T)
    m::T = one(T)
    κ::T = -one(T)
    ν::T = one(T)
    ωx::T = one(T)
    small::T = T(1e-12)
end

phys(cfg::Config) = PhysParams{Float64}(;
    ħ=cfg.ħ, m=cfg.m, κ=cfg.κ, ν=cfg.ν, ωx=cfg.ωx, small=cfg.small
)

# -------------------- k con FFTW.fftfreq --------------------
function k_from_fftfreq_1d(Nx::Int, Lx::Float64)
    dx = Lx / Nx
    kx_cpu = 2π .* FFTW.fftfreq(Nx, 1/dx)
    return CuArray(kx_cpu), dx
end

# -------------------- Potencial 1D --------------------
function harmonic_V_1d(x_gpu::CuArray{Float64}, p::PhysParams{Float64})
    @. 0.5 * p.m * (p.ωx^2) * x_gpu^2
end

# -------------------- Caché 1D --------------------
abstract type SchrCache end

mutable struct SchrCache1D <: SchrCache
    p::PhysParams{Float64}
    x::CuArray{Float64,1}
    kx::CuArray{Float64,1}
    V::CuArray{Float64,1}
    K2::CuArray{Float64,1}
    tmp::CuArray{ComplexF64,1}
    tmp2::CuArray{ComplexF64,1}       # NUEVO: segundo buffer complejo
    ψk::CuArray{ComplexF64,1}
    ρ::CuArray{Float64,1}
    lnρ::CuArray{Float64,1}
    Λ::CuArray{Float64,1}
    ΔΛ::CuArray{Float64,1}
    Λu::CuArray{Float64,1}            # Λ_unwrapped = 2θ
    planF
    planB
    cT::Float64
    Lx::Float64
    dx::Float64
    inv_ik::CuArray{ComplexF64,1}     # 1/(ik), k=0 -> 0
    rho_floor::Float64
    unwrap_kc::Float64
    # stats para JVP
    Z::Float64
    S1::Float64
    μ1::Float64
    SΛ::Float64
    μΛ::Float64
    stats_ready::Bool
end

function make_cache_1d(cfg::Config; Vfun=harmonic_V_1d)
    kx, dx = k_from_fftfreq_1d(cfg.Nx, cfg.Lx)
    x_cpu  = collect(range(-cfg.Lx/2, stop=cfg.Lx/2 - dx, length=cfg.Nx))
    x      = CuArray(x_cpu)
    p      = phys(cfg)
    V      = Vfun(x, p)
    K2     = kx.^2
    tmp    = CUDA.zeros(ComplexF64, cfg.Nx)
    tmp2   = similar(tmp)                     # segundo buffer
    ψk     = similar(tmp)
    ρ      = CUDA.zeros(Float64, cfg.Nx)
    lnρ    = similar(ρ);  Λ = similar(ρ);  ΔΛ = similar(ρ); Λu = similar(ρ)
    planF  = plan_fft(ψk)                     # planes out-of-place
    planB  = plan_ifft(ψk)
    cT     = (p.ħ^2) / (2p.m)

    # 1/(ik) en CPU → GPU (k=0 -> 0)
    inv_ik_cpu = ComplexF64[ ki==0.0 ? 0.0+0.0im : ComplexF64(0,-1)/ki for ki in Array(kx) ]
    inv_ik     = CuArray(inv_ik_cpu)

    return SchrCache1D(p, x, kx, V, K2, tmp, tmp2, ψk, ρ, lnρ, Λ, ΔΛ, Λu,
                       planF, planB, cT, cfg.Lx, dx,
                       inv_ik, cfg.rho_floor, cfg.unwrap_kc,
                       0.0, 0.0, 0.0, 0.0, 0.0, false)
end

# -------------------- Operador H (explícito) --------------------
function Hmul!(y::CuArray{ComplexF64,1}, v::CuArray{ComplexF64,1}, C::SchrCache1D)
    mul!(C.ψk, C.planF, v)                  # ψk = FFT(v)
    @. C.ψk = C.cT * C.K2 * C.ψk            # en k
    mul!(C.tmp, C.planB, C.ψk)              # tmp = IFFT
    @. y = C.tmp + C.V * v                  # y = T v + V v
    return y
end

# f_expl(u) = -(i/ħ) H u
function schrodinger_rhs!(du::CuArray{ComplexF64,1},
                          u::CuArray{ComplexF64,1},
                          C::SchrCache1D, t::Float64)
    Hmul!(du, u, C)
    @. du = ComplexF64(0.0, -1.0) / C.p.ħ * du
    return nothing
end

# -------------------- Unwrap espectral (1D, GPU) --------------------
# θ_unw = F^{-1}{ F[g] * (1/(ik)) },   g = Im(ψ_x / (ψ+ε)); máscara y LP opcional.
function unwrap_phase_spectral_1d!(
    θ_unw::CuArray{Float64,1},
    u::CuArray{ComplexF64,1},
    C::SchrCache1D; eps::Float64 = 1e-14
)
    # ψ_x por FFT: u -> ψk -> tmp (out ≠ in)
    mul!(C.ψk, C.planF, u)
    @. C.ψk = ComplexF64(0,1) * C.kx * C.ψk
    mul!(C.tmp, C.planB, C.ψk)                 # tmp := ∂x ψ

    # g = Im( (∂x ψ)/(ψ+ε) ), con máscara
    @. C.ρ = abs2(u)
    @. C.tmp = C.tmp / (u + Complex(eps,0))
    @. C.lnρ = ifelse(C.ρ > C.rho_floor, imag(C.tmp), 0.0)   # lnρ := g (real)

    # θ̂ = ĝ * 1/(ik)
    @. C.tmp2 = Complex(C.lnρ, 0.0)            # usar tmp2 para FWD
    mul!(C.ψk, C.planF, C.tmp2)
    if isfinite(C.unwrap_kc)
        kc = C.unwrap_kc
        @. C.ψk = C.ψk * exp( - (C.kx/kc)^2 )  # low-pass gaussiano opcional
    end
    @. C.tmp2 = C.ψk * C.inv_ik
    mul!(C.tmp, C.planB, C.tmp2)               # tmp := θ
    @. θ_unw = real(C.tmp)
    return θ_unw
end

# --------- δ (unwrap de fase) coherente con el operador espectral ----------
# δθ = F^{-1}{ (1/(ik)) * F[ δg ] }, con δg = Im( (∂x v)/(u+ε) - (∂x u) * v /(u+ε)^2 )
# δΛu = 2 δθ
function delta_unwrap_phase_spectral_1d!(
    ΔΛ::CuArray{Float64,1},                # salida: δΛu
    v::CuArray{ComplexF64,1},
    u::CuArray{ComplexF64,1},
    C::SchrCache1D; eps::Float64 = 1e-14
)
    # 1) ∂x u  -> tmp
    mul!(C.ψk,  C.planF, u)
    @.  C.ψk = ComplexF64(0,1) * C.kx * C.ψk
    mul!(C.tmp, C.planB, C.ψk)                            # tmp := ∂x u

    # 2) ∂x v  -> tmp2
    mul!(C.ψk,  C.planF, v)
    @.  C.ψk = ComplexF64(0,1) * C.kx * C.ψk
    mul!(C.tmp2, C.planB, C.ψk)                           # tmp2 := ∂x v

    # 3) δg (real) -> lnρ
    @. C.lnρ = imag( C.tmp2 / (u + Complex(eps,0)) -
                     C.tmp  * (v / ((u + Complex(eps,0))*(u + Complex(eps,0)))) )

    # 4) δθ = F^{-1}[ F[δg] * (1/(ik)) ] con low-pass opcional
    @. C.tmp2 = Complex(C.lnρ, 0.0)                       # tmp2 := δg (complex)
    mul!(C.ψk,  C.planF, C.tmp2)                          # ψk := F[δg]
    if isfinite(C.unwrap_kc)
        kc = C.unwrap_kc
        @. C.ψk = C.ψk * exp( - (C.kx/kc)^2 )
    end
    @. C.tmp2 = C.ψk * C.inv_ik                           # tmp2 := F[δg]/(ik)
    mul!(C.tmp,  C.planB, C.tmp2)                         # tmp := δθ
    @. ΔΛ = 2.0 * real(C.tmp)                             # δΛu = 2 δθ
    return ΔΛ
end

# -------------------- Decoherencia (implícita) --------------------
# Wκ = -κ( ln(ρ+ε) - ⟨ln(ρ+ε)⟩_ρ )
# Wν = -i*(ν/2)*( Λ - ⟨Λ⟩_ρ ),    Λ = 2 * θ_unwrapped
function decoherence!(du::CuArray{ComplexF64,1},
                      u::CuArray{ComplexF64,1},
                      C::SchrCache1D, t::Float64)
    p = C.p; ε = p.small

    # ρ, Z
    @. C.ρ = abs2(u)
    Z = sum(C.ρ)

    # f1 = ln(ρ+ε), μ1
    @. C.lnρ = log(C.ρ + ε)
    S1 = sum(@. C.ρ * C.lnρ)
    μ1 = S1 / Z

    # Λ_unwrapped = 2 θ_unw  (guardado para JVP)
    unwrap_phase_spectral_1d!(C.Λu, u, C; eps=ε)  # Λu := θ
    @. C.Λu = 2.0 * C.Λu                          # Λu := 2θ
    SΛ = sum(@. C.ρ * C.Λu)
    μΛ = SΛ / Z

    # W = Wκ + Wν
    @. C.tmp = -p.κ * (C.lnρ - μ1) - ComplexF64(0,1) * (p.ν/2) * (C.Λu - μΛ)
    @. du = C.tmp * u

    # stats para JVP
    C.Z = Z; C.S1 = S1; C.μ1 = μ1; C.SΛ = SΛ; C.μΛ = μΛ; C.stats_ready = true
    return nothing
end

# -------------------- JVP de la parte implícita (decoherencia) --------------------
# Recalcula SIEMPRE stats del u actual (evita desincronización).
@inline function ensure_deco_stats!(u::CuArray{ComplexF64,1}, C::SchrCache1D)
    decoherence!(C.tmp, u, C, 0.0)  # du dummy: sólo poblar stats
    return nothing
end

# Jv = (δWκ+δWν)⊙u + W⊙v
function jvp_decoherence!(Jv::CuArray{ComplexF64,1},
                          v::CuArray{ComplexF64,1},
                          u::CuArray{ComplexF64,1},
                          C::SchrCache1D, t::Float64)
    p = C.p; ε = p.small

    # --- refrescar SIEMPRE las stats del estado actual u ---
    ensure_deco_stats!(u, C)
    Z, S1, μ1, SΛ, μΛ = C.Z, C.S1, C.μ1, C.SΛ, C.μΛ

    # W⊙v
    @. Jv = (-p.κ * (C.lnρ - μ1) - ComplexF64(0,1)*(p.ν/2)*(C.Λu - μΛ)) * v

    # δρ, δZ  (C.Λ := δρ)
    @. C.Λ = 2.0 * real(conj(u) * v)
    δZ = sum(C.Λ)

    # -------- δWκ ⊙ u --------
    # δlnρ = δρ / (ρ+ε)  -> C.ΔΛ
    @. C.ΔΛ = C.Λ / (C.ρ + ε)
    # δS1 = ∑ δρ (lnρ+1) = ∑ (δρ*lnρ + ρ*δlnρ)
    δS1 = sum(@. C.Λ * C.lnρ + C.ρ * C.ΔΛ)
    δμ1 = (δS1 * Z - S1 * δZ) / (Z * Z)
    @. Jv = Jv + (-p.κ * (C.ΔΛ - δμ1)) * u

    # -------- δWν ⊙ u --------
    # δΛu (espectral coherente)
    delta_unwrap_phase_spectral_1d!(C.ΔΛ, v, u, C; eps=ε)  # C.ΔΛ := δΛu
    δSΛ = sum(@. C.Λ * C.Λu + C.ρ * C.ΔΛ)
    δμΛ = (δSΛ * Z - SΛ * δZ) / (Z * Z)
    @. Jv = Jv - ComplexF64(0,1) * (p.ν/2) * (C.ΔΛ - δμΛ) * u

    return nothing
end

# -------------------- Estados coherentes (para u0) --------------------
function coherent1D_gpu(x::CuArray{Float64,1}, α0::ComplexF64, σ::Float64, ω::Float64, t::Float64)
    αt  = α0 * exp(-1im*(ω*t - σ))
    x̄  = sqrt(2/ω) * real(αt)
    p̄  = sqrt(2*ω) * imag(αt)
    Δx  = sqrt(1/(2*ω))
    θ   = -ω*t/2 + (abs2(α0)*sin(2*ω*t - 2σ))/2
    φ   = exp(1im*θ)
    pref= (ω/π)^(1/4)
    den = (2*Δx)^2
    ψ = @. ComplexF64(pref) * ComplexF64(φ) * exp(-((x - x̄)^2)/den) * cis(p̄ * x)
    return ψ
end

# -------------------- Build SplitProblem (decoherencia implícita) --------------------
function build_problem_1d(cfg::Config; Vfun=harmonic_V_1d, ψ0=nothing)
    C = make_cache_1d(cfg; Vfun)
    u0 = ψ0 === nothing ? coherent1D_gpu(C.x, cfg.α0x, cfg.σx, C.p.ωx, cfg.tspan[1]) :
                          CuArray{ComplexF64}(ψ0)

    # EXPLÍCITO: Schrödinger
    f_expl! = (du,u,p,t) -> schrodinger_rhs!(du, u, C, t)
    F_expl  = ODEFunction(f_expl!)

    # IMPLÍCITO (STIFF): Decoherencia + JVP propio
    f_impl! = (du,u,p,t) -> decoherence!(du, u, C, t)
    jvp!    = (Jv,v,u,p,t) -> jvp_decoherence!(Jv, v, u, C, t)
    F_impl  = ODEFunction(f_impl!; jvp=jvp!)

    prob    = SplitODEProblem(F_impl, F_expl, u0, cfg.tspan, C)
    return prob, C
end

# -------------------- Solve (KenCarp58 + GMRES, sin autodiff) --------------------
function solve_problem(cfg::Config; Vfun=harmonic_V_1d, ψ0=nothing)
    prob, cache = build_problem_1d(cfg; Vfun, ψ0)

    # clave: sin autodiff (evita AD en GPU); matrix-free GMRES con jvp
    alg  = KenCarp58(linsolve=KrylovJL_GMRES(), autodiff=false)

    common = (reltol=cfg.reltol, abstol=cfg.abstol, saveat=cfg.saveat,
              maxiters=cfg.maxiters, progress=false)

    if isnothing(cfg.dt) && isnothing(cfg.dtmax)
        sol = solve(prob, alg; common...)
    elseif isnothing(cfg.dt) && !isnothing(cfg.dtmax)
        sol = solve(prob, alg; common..., dtmax=cfg.dtmax)
    elseif !isnothing(cfg.dt) && isnothing(cfg.dtmax)
        sol = solve(prob, alg; common..., dt=cfg.dt)
    else
        sol = solve(prob, alg; common..., dt=cfg.dt, dtmax=cfg.dtmax)
    end

    # Log seguro
    stats = sol.destats
    nsteps   = hasproperty(stats, :nsteps)   ? stats.nsteps   : missing
    naccept  = hasproperty(stats, :naccept)  ? stats.naccept  : missing
    nreject  = hasproperty(stats, :nreject)  ? stats.nreject  : missing
    nfevals  = hasproperty(stats, :nfevals)  ? stats.nfevals  : missing
    @info "retcode=$(sol.retcode) | nsteps=$(nsteps) naccept=$(naccept) nreject=$(nreject) nfevals=$(nfevals)"

    return sol, cache
end

# -------------------- Main --------------------
function main(; kwargs...)
    cfg = Config(; kwargs...)
    sol, cache = solve_problem(cfg)
    @info "Listo. Estados guardados: $(length(sol.t))  |  t_final = $(sol.t[end])"
    return sol, cache
end

# ==================== Utilidades para el ancho δ(t) ====================
# Varianza ponderada por ρ = |u|^2: μ = ⟨x⟩_ρ, var = ⟨(x-μ)^2⟩_ρ, δ = sqrt(var)
function width_gpu!(C::SchrCache1D, u::CuArray{ComplexF64,1})::Float64
    @. C.ρ = abs2(u)
    Z  = sum(C.ρ)
    μ  = sum(@. C.ρ * C.x) / Z
    @. C.Λ = (C.x - μ)^2
    var = sum(@. C.ρ * C.Λ) / Z
    return sqrt(var)
end

function widths_from_solution(sol, cache::SchrCache1D)::Vector{Float64}
    δs = Vector{Float64}(undef, length(sol.t))
    @inbounds for k in eachindex(sol.t)
        δs[k] = width_gpu!(cache, sol.u[k])
    end
    return δs
end

# ODE reducido del ancho (para contraste)
function diff_width!(du, u, p, t)
    δ, dδ = u
    ħ, m, κ, ν, ωx = p
    du[1] = dδ
    du[2] = (2*κ - ν)*dδ + (ν*κ - κ^2)*δ + (ħ^2) / (4*m^2*δ^3) - ωx^2*δ
end

function solve_width_ode(cache::SchrCache1D, tspan::Tuple{Float64,Float64};
                         saveat_times::AbstractVector{<:Real},
                         reltol::Float64=1e-9, abstol::Float64=1e-9)
    pδ = (cache.p.ħ, cache.p.m, cache.p.κ, cache.p.ν, cache.p.ωx)
    δ0   = sqrt(1/(2*cache.p.ωx))
    dδ0  = cache.p.κ * δ0
    u0δ  = [δ0, dδ0]
    sol_δ  = solve(ODEProblem(diff_width!, u0δ, tspan, pδ), AutoVern9(Rodas5P());
                   reltol=reltol, abstol=abstol, saveat=saveat_times)
    return sol_δ
end

# -------------------- Ejemplo mínimo --------------------
# Consejo: empieza sin plotting ni dtmax; luego reintroduce dtmax si quieres.
sol, cache = main(dims=1, Nx=256, Lx=30.0, κ=-1.0, ν=1.0,
                  tspan=(0.0, 1.0), saveat=0.01,
                  unwrap_kc=Inf)  # prueba luego kc = maximum(abs.(Array(cache.kx)))/4

# δ(t) simulado (GPU)
δ_sim = widths_from_solution(sol, cache)

# δ(t) "analítico" (ODE reducido) en los mismos tiempos
sol_δ = solve_width_ode(cache, (sol.t[1], sol.t[end]); saveat_times=sol.t)
δ_ana = first.(sol_δ.u)

# ---- Plot opcional ----
# using GLMakie
# fig = Figure()
# ax  = Axis(fig[1,1], xlabel="Time", ylabel="Width δ(t)")
# lines!(ax, sol.t, δ_sim, label="Simulado", linestyle=:dash)
# lines!(ax, sol_δ.t, δ_ana, label="Analítico (reducido)")
# axislegend(ax)
# display(fig)
