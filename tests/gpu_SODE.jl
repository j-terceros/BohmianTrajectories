##############################
# Schrödinger + Decoherencia #
# SplitODE + GPU (1D/2D)     #
# Precisión: Float64         #
##############################

# -------------------- 0) IMPORTS (GPU + ODE + FFT) --------------------
using CUDA                      # CuArray y ejecución en GPU
using CUDA.CUFFT                # planes FFT en GPU (cuFFT)
using LinearAlgebra             # norm, etc.
using DifferentialEquations     # SplitODEProblem y solve
using OrdinaryDiffEq            # algoritmos IMEX (KenCarp*)
using LinearSolve               # GMRES (KrylovJL_GMRES) para JFNK
using FFTW                      # fftfreq (tu forma preferida para k)
using GLMakie                   # visualización (opcional)

CUDA.allowscalar(false)         # prohíbe indexado escalar en GPU

# -------------------- 1) CONFIGURACIÓN GLOBAL (0% hard-coded) --------------------
"""
Config: toda la configuración del problema.
- dims: 1 ó 2 (dimensión espacial)
- Nx, Ny: puntos de malla (Ny se ignora si dims=1)
- Lx, Ly: longitudes del dominio (Ly se ignora si dims=1)
- tspan: intervalo temporal
- ħ, m: constantes físicas
- κ, ν: parámetros de decoherencia
- ωx, ωy: frecuencias del potencial armónico (ωy se usa solo en 2D)
- small: regularizador para log(ρ+small)
- reltol, abstol: tolerancias del solver
- saveat: cadencia de guardado de solución
- dt: paso fijo opcional (si nothing, paso adaptativo)
- krylovdim: dimensión subespacio GMRES (afecta RAM)
"""
Base.@kwdef struct Config
    dims::Int                         = 1
    Nx::Int                           = 1024
    Ny::Int                           = 1024
    Lx::Float64                       = 30.0
    Ly::Float64                       = 30.0
    tspan::Tuple{Float64,Float64}     = (0.0, 5.0)
    ħ::Float64                        = 1.0
    m::Float64                        = 1.0
    κ::Float64                        = 0.0
    ν::Float64                        = 0.0
    ωx::Float64                       = 1.0
    ωy::Float64                       = 1.0
    small::Float64                    = 1e-16
    reltol::Float64                   = 1e-12
    abstol::Float64                   = 1e-12
    saveat::Float64                   = 0.01
    dt::Union{Nothing,Float64}        = nothing
    krylovdim::Int                    = 20
    # Parámetros del estado coherente y entrelazado
    α0x::ComplexF64                   = 2.5 + 0.0im    # amplitud coherente eje x
    σx::Float64                       = 0.0            # fase inicial eje x
    α0y::ComplexF64                   = 2.5 + 0.0im    # amplitud coherente eje y (2D)
    σy::Float64                       = 0.0            # fase inicial eje y (2D)
    c1::ComplexF64                    = 1.0 + 0.0im    # coeficiente del término activo en 2D
    c2::ComplexF64                    = 0.0 + 0.0im    # (reservado) segundo término si luego lo activas
end

# empaquetado de parámetros físicos (cómodo para pasar en cache)
Base.@kwdef struct PhysParams{T}
    ħ::T = one(T)
    m::T = one(T)
    κ::T = -one(T)
    ν::T = one(T)
    ωx::T = one(T)
    ωy::T = one(T)
    small::T = T(1e-16)
end

phys(cfg::Config) = PhysParams{Float64}(;
    ħ=cfg.ħ, m=cfg.m, κ=cfg.κ, ν=cfg.ν, ωx=cfg.ωx, ωy=cfg.ωy, small=cfg.small
)

# -------------------- 2) FRECUENCIAS k CON FFTW.fftfreq (tu preferencia) --------------------
# Usamos tu forma: k = 2π * fftfreq(N, 1/dx) → rad/m en el orden correcto.
# Se calcula en CPU (FFTW) y subimos a GPU una sola vez.

function k_from_fftfreq_1d(Nx::Int, Lx::Float64)
    dx = Lx / Nx
    kx_cpu = 2π .* FFTW.fftfreq(Nx, 1/dx)   # Vector{Float64} en CPU
    return CuArray(kx_cpu), dx               # pasamos a GPU y devolvemos dx
end

function k_from_fftfreq_2d(Nx::Int, Ny::Int, Lx::Float64, Ly::Float64)
    dx = Lx / Nx; dy = Ly / Ny
    kx_cpu = 2π .* FFTW.fftfreq(Nx, 1/dx)
    ky_cpu = 2π .* FFTW.fftfreq(Ny, 1/dy)
    return CuArray(kx_cpu), CuArray(ky_cpu), dx, dy
end

# -------------------- 3) POTENCIAL ARMÓNICO (puedes inyectar otro) --------------------
# 1D: V(x) = 1/2 m ωx^2 x^2
function harmonic_V_1d(x_gpu::CuArray{Float64}, p::PhysParams{Float64})
    @. 0.5 * p.m * (p.ωx^2) * x_gpu^2
end

# 2D: V(x,y) = 1/2 m (ωx^2 x^2 + ωy^2 y^2)
function harmonic_V_2d(x_gpu::CuArray{Float64}, y_gpu::CuArray{Float64}, p::PhysParams{Float64})
    Nx = length(x_gpu); Ny = length(y_gpu)
    X2 = reshape(x_gpu.^2, 1, Nx)           # 1×Nx (broadcast-friendly)
    Y2 = reshape(y_gpu.^2, Ny, 1)           # Ny×1
    @. 0.5 * p.m * (p.ωx^2 * X2 + p.ωy^2 * Y2)   # Ny×Nx en GPU
end

# -------------------- 4) CACHÉS (planes FFT + buffers + constantes) --------------------
abstract type SchrCache end

# 1D
struct SchrCache1D <: SchrCache
    p::PhysParams{Float64}                 # parámetros físicos
    x::CuArray{Float64,1}                  # malla x
    kx::CuArray{Float64,1}                 # frecuencias kx
    V::CuArray{Float64,1}                  # potencial V(x)
    K2::CuArray{Float64,1}                 # kx.^2
    tmp::CuArray{ComplexF64,1}             # buffer real-space
    ψk::CuArray{ComplexF64,1}              # buffer k-space
    ρ::CuArray{Float64,1}                  # densidad |ψ|^2
    lnρ::CuArray{Float64,1}                # log(ρ+ε)
    Λ::CuArray{Float64,1}                  # 2*arg(ψ)
    ΔΛ::CuArray{Float64,1}                 # desviación angular envuelta
    planF                                   # plan FFT (no anotamos tipo)
    planB                                   # plan IFFT (no anotamos tipo)
    cT::Float64                            # ħ^2/(2m)
    Lx::Float64                            # longitud del dominio
    dx::Float64                            # paso en x (evita indexado escalar)
end

# 2D
struct SchrCache2D <: SchrCache
    p::PhysParams{Float64}
    x::CuArray{Float64,1}
    y::CuArray{Float64,1}
    kx::CuArray{Float64,1}
    ky::CuArray{Float64,1}
    V::CuArray{Float64,2}
    K2::CuArray{Float64,2}                 # ky.^2 .+ kx.^2 (Ny×Nx)
    tmp::CuArray{ComplexF64,2}
    ψk::CuArray{ComplexF64,2}
    ρ::CuArray{Float64,2}
    lnρ::CuArray{Float64,2}
    Λ::CuArray{Float64,2}
    ΔΛ::CuArray{Float64,2}
    planF
    planB
    cT::Float64
    Lx::Float64; Ly::Float64
    dx::Float64; dy::Float64               # pasos (sin indexar)
end

# crea cache 1D (potencial inyectable)
function make_cache_1d(cfg::Config; Vfun=harmonic_V_1d)
    kx, dx = k_from_fftfreq_1d(cfg.Nx, cfg.Lx)
    x_cpu  = collect(range(-cfg.Lx/2, stop=cfg.Lx/2 - dx, length=cfg.Nx)) # CPU
    x      = CuArray(x_cpu)                                               # GPU
    p      = phys(cfg)
    V      = Vfun(x, p)
    K2     = kx.^2
    tmp    = CUDA.zeros(ComplexF64, cfg.Nx)
    ψk     = similar(tmp)
    ρ      = CUDA.zeros(Float64, cfg.Nx)
    lnρ    = similar(ρ);  Λ = similar(ρ);  ΔΛ = similar(ρ)
    planF  = plan_fft(ψk)                  # cuFFT 1D
    planB  = plan_ifft(ψk)
    cT     = (p.ħ^2) / (2p.m)
    SchrCache1D(p, x, kx, V, K2, tmp, ψk, ρ, lnρ, Λ, ΔΛ, planF, planB, cT, cfg.Lx, dx)
end

# crea cache 2D
function make_cache_2d(cfg::Config; Vfun=harmonic_V_2d)
    kx, ky, dx, dy = k_from_fftfreq_2d(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
    x_cpu = collect(range(-cfg.Lx/2, stop=cfg.Lx/2 - dx, length=cfg.Nx))
    y_cpu = collect(range(-cfg.Ly/2, stop=cfg.Ly/2 - dy, length=cfg.Ny))
    x     = CuArray(x_cpu);  y = CuArray(y_cpu)
    p     = phys(cfg)
    KX2   = reshape(kx.^2, 1, cfg.Nx)      # 1×Nx
    KY2   = reshape(ky.^2, cfg.Ny, 1)      # Ny×1
    K2    = @. KY2 + KX2                   # Ny×Nx
    V     = Vfun(x, y, p)
    tmp   = CUDA.zeros(ComplexF64, cfg.Ny, cfg.Nx)
    ψk    = similar(tmp)
    ρ     = CUDA.zeros(Float64, cfg.Ny, cfg.Nx)
    lnρ   = similar(ρ);  Λ = similar(ρ);  ΔΛ = similar(ρ)
    planF = plan_fft(ψk)                   # cuFFT 2D
    planB = plan_ifft(ψk)
    cT    = (p.ħ^2) / (2p.m)
    SchrCache2D(p, x, y, kx, ky, V, K2, tmp, ψk, ρ, lnρ, Λ, ΔΛ, planF, planB, cT, cfg.Lx, cfg.Ly, dx, dy)
end

# -------------------- 5) OPERADOR H (T + V) EN GPU, vía FFT --------------------
# Tψ = ℱ⁻¹( (ħ²/2m) K² .* ℱ(ψ) ); Vψ = V .* ψ.  cuFFT no normaliza → multiplicamos por invN.

# 1D
function Hmul!(y, v, C::SchrCache1D)
    # FFT out-of-place: ψk = FFT(v)
    mul!(C.ψk, C.planF, v)

    # aplica cinética en k: ψk .= cT * K2 .* ψk
    @. C.ψk = C.cT * C.K2 * C.ψk

    # IFFT out-of-place: tmp = IFFT(ψk)
    mul!(C.tmp, C.planB, C.ψk)
    #@. C.tmp = C.invN * C.tmp  # normaliza (cuFFT no normaliza)

    # y = T v + V v
    @. y = C.tmp + C.V * v
    return y
end


# 2D
function Hmul!(y, v, C::SchrCache2D)
    mul!(C.ψk, C.planF, v)          # ψk = FFT2(v)
    @. C.ψk = C.cT * C.K2 * C.ψk    # cinética en k
    mul!(C.tmp, C.planB, C.ψk)      # tmp = IFFT2(ψk)
    #@. C.tmp = C.invN * C.tmp        # normaliza
    @. y = C.tmp + C.V * v           # y = T v + V v
    return y
end


# -------------------- 6) f₁(u) = -(i/ħ) H u  (parte implícita, lineal) --------------------
# Única función que sirve para 1D y 2D: el múltiple dispatch de Hmul! resuelve.
function schrodinger_impl!(du, u, C::SchrCache, t)
    Hmul!(du, u, C)                              # du := H u
    @. du = ComplexF64(0, -1) / C.p.ħ * du       # du := -(i/ħ) du
    return nothing
end

# -------------------- 7) f₂(u) = decoherencia (parte explícita) --------------------
# ΔΛ envuelto a [-π, π] (evita saltos angulares grandes)
wrap_to_pi!(out, x) = (@. out = (mod(x + π, 2π)) - π)

function decoherence!(du, u, C::SchrCache, t)
    p = C.p
    @. C.ρ   = abs2(u)                      # ρ = |ψ|^2
    Z = sum(C.ρ)                            # normalización discreta (GPU→host escalar)

    @. C.lnρ = log(C.ρ + p.small)           # lnρ = log(ρ + ε)
    μln = sum(@. C.ρ * C.lnρ) / Z           # media ponderada por ρ

    @. C.Λ   = 2.0 * angle(u)               # ángulo doble
    s = sum(@. C.ρ * sin(C.Λ))              # suma ponderada de senos
    c = sum(@. C.ρ * cos(C.Λ))              # suma ponderada de cosenos
    μΛ = atan(s, c)                          # media circular

    wrap_to_pi!(C.ΔΛ, @. C.Λ - μΛ)          # ΔΛ ∈ [-π, π]

    # du = [ -κ(lnρ - <lnρ>) - i*(ν/2)*ΔΛ ] * u
    @. du = ( -p.κ*(C.lnρ - μln) - ComplexF64(0,1)*(p.ν/2)*C.ΔΛ ) * u
    return nothing
end

# -------------------- 8) jvp: producto direccional J*v (sin autodiff, ni jacobiano) ------
# Para f_impl(u) = -(i/ħ) (T u + V.*u), su derivada es lineal: Jv = -(i/ħ) (T v + V.*v)
# Lo implementamos en GPU. OrdinaryDiffEq lo usará para formar Wv = v - γ Jv en GMRES.

# 1D
function jvp_schrodinger!(Jv::CuArray{ComplexF64,1}, v::CuArray{ComplexF64,1}, C::SchrCache1D)
    mul!(C.ψk, C.planF, v)          # ψk = FFT(v)
    @. C.ψk = C.cT * C.K2 * C.ψk    # cinética en k
    mul!(C.tmp, C.planB, C.ψk)      # tmp = IFFT(ψk)
    #@. C.tmp = C.invN * C.tmp
    @. C.tmp = C.tmp + C.V * v
    @. Jv = ComplexF64(0, -1) / C.p.ħ * C.tmp
    return nothing
end

# 2D
function jvp_schrodinger!(Jv::CuArray{ComplexF64,2}, v::CuArray{ComplexF64,2}, C::SchrCache2D)
    mul!(C.ψk, C.planF, v)          # ψk = FFT2(v)
    @. C.ψk = C.cT * C.K2 * C.ψk
    mul!(C.tmp, C.planB, C.ψk)      # tmp = IFFT2(ψk)
    #@. C.tmp = C.invN * C.tmp
    @. C.tmp = C.tmp + C.V * v
    @. Jv = ComplexF64(0, -1) / C.p.ħ * C.tmp
    return nothing
end


# -------------------- 9) ESTADOS INICIALES (sin indexado escalar; usan dx/dy del cache) ---
# Gauss fundamental del oscilador armónico (puedes reemplazar por tu coherente si quieres)

# Estado coherente 1D (GPU, sin indexado escalar)
# Mantiene exactamente tu forma: exp( -(x- x̄)^2 / (2*Δx)^2 )  (den = (2Δx)^2)
function coherent1D_gpu(x::CuArray{Float64,1}, α0::ComplexF64, σ::Float64, ω::Float64, t::Float64)
    αt  = α0 * exp(-1im*(ω*t - σ))
    x̄  = sqrt(2/ω) * real(αt)
    p̄  = sqrt(2*ω) * imag(αt)
    Δx  = sqrt(1/(2*ω))
    θ   = -ω*t/2 + (abs2(α0)*sin(2*ω*t - 2σ))/2     # abs2 == |α0|^2
    φ   = exp(1im*θ)
    pref= (ω/π)^(1/4)
    den = (2*Δx)^2                                  # <- así reproducimos tu denominador

    # ψ(x,t) = pref * φ * exp(-((x-x̄)^2)/den) * exp(i p̄ x)
    ψ = @. ComplexF64(pref) * ComplexF64(φ) *
              exp(-((x - x̄)^2)/den) *
              cis(p̄ * x)                           # cis(z)=exp(i z), estable y GPU-friendly
    return ψ                                        # NO normalizamos aquí
end

# Estado entrelazado 2D (GPU): solo el término c1 · ψ_rx(x) · ψ_ry(y)
# Devuelve Ny×Nx (primera dim y), consistente con V, K2, ψk, tmp del caché.
function entangled_ψ_gpu(x::CuArray{Float64,1}, y::CuArray{Float64,1},
                         α0x::ComplexF64, σx::Float64, ωx::Float64,
                         α0y::ComplexF64, σy::Float64, ωy::Float64,
                         t::Float64, c1::ComplexF64, c2::ComplexF64)
    # Coherentes 1D en cada eje (sin normalizar)
    ψrx = coherent1D_gpu(x, α0x, σx, ωx, t)    # Nx
    ψry = coherent1D_gpu(y, α0y, σy, ωy, t)    # Ny

    Nx = length(x); Ny = length(y)

    # Construye Ny×Nx = ψ_ry(y) * ψ_rx(x) (outer product con broadcasting)
    # reshape(ψry, Ny,1) .* reshape(ψrx,1,Nx) → Ny×Nx
    ψ = @. ComplexF64(c1) * reshape(ψry, Ny, 1) * reshape(ψrx, 1, Nx)

    # Si más adelante activas el segundo término (comentado en tu idea):
    # ψlx = coherent1D_gpu(x, α0x, σx+π, ωx, t)
    # ψly = coherent1D_gpu(y, α0y, σy+π, ωy, t)
    # ψ  += @. ComplexF64(c2) * reshape(ψly, Ny, 1) * reshape(ψrx, 1, Nx) # o la combinación que quieras

    return ψ
end

# -------------------- 10) CONSTRUCCIÓN DEL SPLIT PROBLEM (1D / 2D) -----------------------
# Nota: ODEFunction con keyword :jvp (NO Wmul/Wfact). Esto evita autodiff en GPU.

function build_problem_1d(cfg::Config; Vfun=harmonic_V_1d, ψ0=nothing)
    C = make_cache_1d(cfg; Vfun)
    # Estado inicial:
    if ψ0 === nothing
        ψ = coherent1D_gpu(C.x, cfg.α0x, cfg.σx, C.p.ωx, cfg.tspan[1])     # t0
        # Normalización discreta: ∑ |ψ|^2 dx = 1
        #norm2 = sum(abs2.(ψ)) * C.dx
        #ψ ./= sqrt(norm2)
    else
        ψ = CuArray{ComplexF64}(ψ0)
    end

    f_impl! = (du,u,p,t) -> schrodinger_impl!(du, u, C, t)
    jvp!    = (Jv,v,u,p,t) -> jvp_schrodinger!(Jv, v, C)
    f_expl! = (du,u,p,t) -> decoherence!(du, u, C, t)

    F_impl  = ODEFunction(f_impl!; jvp=jvp!)
    F_expl  = ODEFunction(f_expl!)
    prob    = SplitODEProblem(F_impl, F_expl, ψ, cfg.tspan, C)
    return prob, C
end

function build_problem_2d(cfg::Config; Vfun=harmonic_V_2d, ψ0=nothing)
    C = make_cache_2d(cfg; Vfun)
    if ψ0 === nothing
        ψ = entangled_ψ_gpu(C.x, C.y,
                            cfg.α0x, cfg.σx, C.p.ωx,
                            cfg.α0y, cfg.σy, C.p.ωy,
                            cfg.tspan[1], cfg.c1, cfg.c2)
        # Normalización discreta 2D: ∑ |ψ|^2 dx dy = 1
        #norm2 = sum(abs2.(ψ)) * C.dx * C.dy
        #ψ ./= sqrt(norm2)
    else
        ψ = CuArray{ComplexF64}(ψ0)
    end

    f_impl! = (du,u,p,t) -> schrodinger_impl!(du, u, C, t)
    jvp!    = (Jv,v,u,p,t) -> jvp_schrodinger!(Jv, v, C)
    f_expl! = (du,u,p,t) -> decoherence!(du, u, C, t)

    F_impl  = ODEFunction(f_impl!; jvp=jvp!)
    F_expl  = ODEFunction(f_expl!)
    prob    = SplitODEProblem(F_impl, F_expl, ψ, cfg.tspan, C)
    return prob, C
end


function build_problem_2d(cfg::Config; Vfun=harmonic_V_2d, ψ0=nothing)
    C = make_cache_2d(cfg; Vfun)
    if ψ0 === nothing
        ψ = entangled_ψ_gpu(C.x, C.y,
                            cfg.α0x, cfg.σx, C.p.ωx,
                            cfg.α0y, cfg.σy, C.p.ωy,
                            cfg.tspan[1], cfg.c1, cfg.c2)
        # Normalización discreta 2D: ∑ |ψ|^2 dx dy = 1
        #norm2 = sum(abs2.(ψ)) * C.dx * C.dy
        #ψ ./= sqrt(norm2)
    else
        ψ = CuArray{ComplexF64}(ψ0)
    end

    f_impl! = (du,u,p,t) -> schrodinger_impl!(du, u, C, t)
    jvp!    = (Jv,v,u,p,t) -> jvp_schrodinger!(Jv, v, C)
    f_expl! = (du,u,p,t) -> decoherence!(du, u, C, t)

    F_impl  = ODEFunction(f_impl!; jvp=jvp!)
    F_expl  = ODEFunction(f_expl!)
    prob    = SplitODEProblem(F_impl, F_expl, ψ, cfg.tspan, C)
    return prob, C
end

# -------------------- 11) ESTIMADOR DE MEMORIA (para evitar OOM) -------------------------
# Estima VRAM ~ base (3 complejos + 6 reales) + Krylov (krylovdim estados) + overhead.

function estimate_memory_bytes(cfg::Config)
    N = cfg.dims == 1 ? cfg.Nx : cfg.Nx * cfg.Ny
    bytes_state      = 16N                    # ψ
    bytes_complexbuf = 2 * 16N               # tmp + ψk
    bytes_realbuf    = 6 * 8N                # V, K2, ρ, lnρ, Λ, ΔΛ
    bytes_krylov     = cfg.krylovdim * 16N   # base GMRES
    bytes_stages     = 4 * 16N               # etapas/intermedios (aprox)
    return bytes_state + bytes_complexbuf + bytes_realbuf + bytes_krylov + bytes_stages
end

# formateo simple sin Printf
function pretty_bytes(b::Real)
    units = ["B","KB","MB","GB","TB","PB"]
    x = float(b); i = 1
    while x >= 1024 && i < length(units)
        x /= 1024; i += 1
    end
    return string(round(x; digits=2), " ", units[i])
end

# -------------------- 12) RESOLVER: KenCarp58 + GMRES (JFNK, sin autodiff) ---------------
function solve_problem(cfg::Config; Vfun=cfg.dims==1 ? harmonic_V_1d : harmonic_V_2d, ψ0=nothing)
    # imprime estimación de memoria antes de correr
    est = estimate_memory_bytes(cfg)
    @info "Estimación de memoria (solo arrays + GMRES base): $(pretty_bytes(est))" cfg

    # construye problema según dimensión
    prob, cache = cfg.dims == 1 ? build_problem_1d(cfg; Vfun, ψ0) :
                                  build_problem_2d(cfg; Vfun, ψ0)

    # IMEX de orden 5 con linsolve=GMRES (matricial-libre usando tu jvp)
    lins = KrylovJL_GMRES()
    alg  = KenCarp58(linsolve=lins)  # sin autodiff; usará jvp

    common = (reltol=cfg.reltol, abstol=cfg.abstol, saveat=cfg.saveat)
    sol = isnothing(cfg.dt) ? solve(prob, alg; common...) :
                              solve(prob, alg; dt=cfg.dt, common...)

    return sol, cache
end

# -------------------- 13) MAIN para VS Code / REPL / CLI ---------------------------------
"""
main(; kwargs...) -> ejecuta con una Config personalizada.
Ejemplos:
  include("gpu_SODE.jl")
  sol1, cache1 = main(dims=1, Nx=1024, Lx=30.0, krylovdim=12)

  sol2, cache2 = main(dims=2, Nx=1024, Ny=1024, Lx=30.0, Ly=30.0, krylovdim=15)
"""
function main(; kwargs...)
    cfg = Config(; kwargs...)
    sol, cache = solve_problem(cfg)
    @info "Listo. Estados guardados: $(length(sol.t))  |  t_final = $(sol.t[end])"
    return sol, cache
end

sol1, cache1 = main(dims=1, Nx=1024, Lx=30.0, κ=0.0, ν=0.0,
                    α0x=2.5+0im, σx=0.0)

# 1) Toma x y ψ_sim del resultado, y pásalos a CPU
x_cpu   = Array(cache1.x)          # cache1 lo devolvió main(...)
ψ_sim   = Array(sol1.u[end])       # último estado en t = tspan[2]
ψ_sim_r = real.(ψ_sim)             # parte real (o usa abs.(ψ_sim) si quieres módulo)

# 2) Define tu coherente analítica en CPU (sin GPU), estilo “tu” fórmula
#    OJO: esta es la evolución SIN decoherencia. Para comparar con precisión,
#    pon κ=0 y ν=0 en Config, o entiende que no coincidirán si hay decoherencia.
# Estado coherente 1D (GPU, sin indexado escalar)
# Mantiene exactamente tu forma: exp( -(x- x̄)^2 / (2*Δx)^2 )  (den = (2Δx)^2)
function coherent1D_cpu(x::AbstractVector{<:Real}, α0::ComplexF64, σ::Float64, ω::Float64, t::Float64)
    αt  = α0 * exp(-1im*(ω*t - σ))
    x̄  = sqrt(2/ω) * real(αt)
    p̄  = sqrt(2*ω) * imag(αt)
    Δx  = sqrt(1/(2*ω))
    θ   = -ω*t/2 + (abs2(α0)*sin(2*ω*t - 2σ))/2
    pref = (ω/π)^(1/4)
    # Gauss con anchura correcta: exp(- (x-x̄)^2 / (2Δx^2)) y fase p̄ x
    return pref .* exp.(1im*θ) .* exp.(-((x .- x̄).^2) ./ (2*Δx)^2) .* exp.(1im*p̄ .* x)
end

# 3) Construye la analítica (elige tus α0 y σ)
α0x = 2.5 + 0.0im     # ej: 5/2 como usabas
σx  = 0.0
ωx  = cache1.p.ωx     # usa la ω del problema
t_f = sol1.t[end]     # último tiempo del numérico
ψf   = coherent1D_cpu(x_cpu, α0x, σx, ωx, t_f)
ψf_r = real.(ψf)

# 4) Grafica
fig = Figure()
ax  = Axis(fig[1,1], xlabel="x", ylabel="Re ψ(x)")
lines!(ax, x_cpu, ψ_sim_r, label="Simulado (GPU)", linestyle=:dash)
lines!(ax, x_cpu, ψf_r,    label="Analítico (coherente)")
axislegend(ax)
display(fig)