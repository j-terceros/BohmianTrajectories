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
Base.@kwdef struct Config{T<:AbstractFloat}
    dims::Int                         = 1
    Nx::Int                           = 1024
    Ny::Int                           = 1024
    Lx::T                       = 30.0
    Ly::T                       = 30.0
    tspan::Tuple{T,T}     = (0.0, 5.0)
    ħ::T                        = 1.0
    m::T                        = 1.0
    κ::T                        = 0.0
    ν::T                        = 0.0
    ωx::T                       = 1.0
    ωy::T                       = 1.0
    small::T                    = eps(T)
    reltol::T                   = eps(T)*1e4
    abstol::T                   = eps(T)*1e4
    saveat::T                   = 0.01
    dt::Union{Nothing,T}        = nothing
    krylovdim::Int                    = 20
    # Parámetros del estado coherente y entrelazado
    α0x::Complex{T}                   = 2.5 + 0.0im    # amplitud coherente eje x
    σx::T                       = 0.0            # fase inicial eje x
    α0y::Complex{T}                   = 2.5 + 0.0im    # amplitud coherente eje y (2D)
    σy::T                       = 0.0            # fase inicial eje y (2D)
    c1::Complex{T}                    = 1.0 + 0.0im    # coeficiente del término activo en 2D
    c2::Complex{T}                    = 0.0 + 0.0im    # (reservado) segundo término si luego lo activas
end

# empaquetado de parámetros físicos (cómodo para pasar en cache)
Base.@kwdef struct PhysParams{T} 
    ħ::T = one(T)
    m::T = one(T)
    κ::T = -one(T)
    ν::T = one(T)
    ωx::T = one(T)
    ωy::T = one(T)
    small::T = eps(T)
end

phys(cfg::Config{T}) where T = PhysParams{T}(;
    ħ=cfg.ħ, m=cfg.m, κ=cfg.κ, ν=cfg.ν, ωx=cfg.ωx, ωy=cfg.ωy, small=cfg.small
)

# -------------------- 2) FRECUENCIAS k CON FFTW.fftfreq (tu preferencia) --------------------
# Usamos tu forma: k = 2π * fftfreq(N, 1/dx) → rad/m en el orden correcto.
# Se calcula en CPU (FFTW) y subimos a GPU una sola vez.

function k_from_fftfreq_1d(Nx, Lx)
    dx = Lx / Nx
    kx = 2π .* FFTW.fftfreq(Nx, 1/dx)   # Vector{Float64} en CPU
    return kx, dx               # no pasamos a GPU y devolvemos dx
end

function k_from_fftfreq_2d(Nx, Ny, Lx, Ly)
    dx = Lx / Nx; dy = Ly / Ny
    kx = 2π .* FFTW.fftfreq(Nx, 1/dx)
    ky = 2π .* FFTW.fftfreq(Ny, 1/dy)
    return kx, ky, dx, dy
end

# -------------------- 3) POTENCIAL ARMÓNICO (puedes inyectar otro) --------------------
# 1D: V(x) = 1/2 m ωx^2 x^2
function harmonic_V_1d(x::AbstractArray{T}, p::PhysParams{T}) where T
    @. 0.5 * p.m * (p.ωx^2) * x^2
end

# 2D: V(x,y) = 1/2 m (ωx^2 x^2 + ωy^2 y^2)
function harmonic_V_2d(x::AbstractArray{T,1}, y::AbstractArray{T,1}, p::PhysParams{T}) where T
    Nx = length(x); Ny = length(y)
    X2 = reshape(x.^2, 1, Nx)           # 1×Nx (broadcast-friendly)
    Y2 = reshape(y.^2, Ny, 1)           # Ny×1
    @. 0.5 * p.m * (p.ωx^2 * X2 + p.ωy^2 * Y2)   # Ny×Nx en GPU
end

# -------------------- 4) CACHÉS (planes FFT + buffers + constantes) --------------------
abstract type SchrCache{T} end

# 1D
struct SchrCache1D{T} <: SchrCache{T}
    p::PhysParams{T}                 # parámetros físicos
    x::AbstractArray{T,1}                  # malla x
    kx::AbstractArray{T,1}                 # frecuencias kx
    V::AbstractArray{T,1}                  # potencial V(x)
    K2::AbstractArray{T,1}                 # kx.^2
    tmp::AbstractArray{Complex{T},1}             # buffer real-space
    ψk::AbstractArray{Complex{T},1}              # buffer k-space
    ρ::AbstractArray{T,1}                  # densidad |ψ|^2
    lnρ::AbstractArray{T,1}                # log(ρ+ε)
    Λ::AbstractArray{T,1}                  # 2*arg(ψ)
    ΔΛ::AbstractArray{T,1}                 # desviación angular envuelta
    planF                                   # plan FFT (no anotamos tipo)
    planB                                   # plan IFFT (no anotamos tipo)
    cT::T                            # ħ^2/(2m)
    Lx::T                            # longitud del dominio
    dx::T                            # paso en x (evita indexado escalar)
end

# 2D
struct SchrCache2D{T} <: SchrCache{T}
    p::PhysParams{T}
    x::AbstractArray{T,1}
    y::AbstractArray{T,1}
    kx::AbstractArray{T,1}
    ky::AbstractArray{T,1}
    V::AbstractArray{T,2}
    K2::AbstractArray{T,2}                 # ky.^2 .+ kx.^2 (Ny×Nx)
    tmp::AbstractArray{Complex{T},2}
    ψk::AbstractArray{Complex{T},2}
    ρ::AbstractArray{T,2}
    lnρ::AbstractArray{T,2}
    Λ::AbstractArray{T,2}
    ΔΛ::AbstractArray{T,2}
    planF
    planB
    cT::T
    Lx::T; Ly::T
    dx::T; dy::T               # pasos (sin indexar)
end

# crea cache 1D (potencial inyectable)
function make_cache_1d(cfg::Config{T}, to_device, Vfun) where T
    kx, dx = k_from_fftfreq_1d(cfg.Nx, cfg.Lx)
    kx = to_device(kx)
    x  = collect(range(-cfg.Lx/2, stop=cfg.Lx/2 - dx, length=cfg.Nx)) # CPU
    x = to_device(x)
    p      = phys(cfg)
    V      = Vfun(x, p)
    K2     = kx.^2
    tmp    = zeros(complex(T), cfg.Nx) |> to_device
    ψk     = similar(tmp)
    ρ      = zeros(T, cfg.Nx) |> to_device
    lnρ    = similar(ρ);  Λ = similar(ρ);  ΔΛ = similar(ρ)
    planF  = plan_fft(ψk)                  # cuFFT 1D
    planB  = plan_ifft(ψk)
    cT     = (p.ħ^2) / (2p.m)
    SchrCache1D{T}(p, x, kx, V, K2, tmp, ψk, ρ, lnρ, Λ, ΔΛ, planF, planB, cT, cfg.Lx, dx)
end

# crea cache 2D
function make_cache_2d(cfg::Config{T}, to_device, Vfun) where T
    kx, ky, dx, dy = k_from_fftfreq_2d(cfg.Nx, cfg.Ny, cfg.Lx, cfg.Ly)
    kx = to_device(kx)
    ky = to_device(ky)
    x = collect(range(-cfg.Lx/2, stop=cfg.Lx/2 - dx, length=cfg.Nx))
    y = collect(range(-cfg.Ly/2, stop=cfg.Ly/2 - dy, length=cfg.Ny))
    x = to_device(x)
    y = to_device(y)
    p     = phys(cfg)
    KX2   = reshape(kx.^2, 1, cfg.Nx)      # 1×Nx
    KY2   = reshape(ky.^2, cfg.Ny, 1)      # Ny×1
    K2    = @. KY2 + KX2                   # Ny×Nx
    V     = Vfun(x, y, p)
    tmp   = zeros(complex(T), cfg.Ny, cfg.Nx) |> to_device
    ψk    = similar(tmp)
    ρ     = zeros(T, cfg.Ny, cfg.Nx) |> to_device
    lnρ   = similar(ρ);  Λ = similar(ρ);  ΔΛ = similar(ρ)
    planF = plan_fft(ψk)                   # cuFFT 2D
    planB = plan_ifft(ψk)
    cT    = (p.ħ^2) / (2p.m)
    SchrCache2D{T}(p, x, y, kx, ky, V, K2, tmp, ψk, ρ, lnρ, Λ, ΔΛ, planF, planB, cT, cfg.Lx, cfg.Ly, dx, dy)
end

# -------------------- 5) OPERADOR H (T + V) EN GPU, vía FFT --------------------
# Tψ = ℱ⁻¹( (ħ²/2m) K² .* ℱ(ψ) ); Vψ = V .* ψ.  cuFFT no normaliza → multiplicamos por invN.

# ND
function Hmul!(y, v, C::SchrCache{T}) where T
    # FFT out-of-place: ψk = FFT(v)
    mul!(C.ψk, C.planF, v)

    # aplica cinética en k: ψk .= cT * K2 .* ψk
    @. C.ψk = C.cT * C.K2 * C.ψk

    # IFFT out-of-place: tmp = IFFT(ψk)
    mul!(C.tmp, C.planB, C.ψk)
    #@. C.tmp = C.invN * C.tmp  # normaliza (cuFFT no normaliza)

    # y = T v + V v
    @. y = C.tmp + C.V * v
    return nothing
end

# -------------------- 6) f₁(u) = -(i/ħ) H u  (parte implícita, lineal) --------------------
# Única función que sirve para 1D y 2D: el múltiple dispatch de Hmul! resuelve.
function schrodinger_impl!(du, u, C::SchrCache{T}, t) where T
    Hmul!(du, u, C)                              # du := H u
    @. du = complex(T)( 0, -1) / C.p.ħ * du       # du := -(i/ħ) du
    return nothing
end

# -------------------- 7) f₂(u) = decoherencia (parte explícita) --------------------
# ΔΛ envuelto a [-π, π] (evita saltos angulares grandes)
wrap_to_pi!(out, x) = (@. out = (mod(x + π, 2π)) - π)

function decoherence!(du, u, C::SchrCache{T}, t) where T
    p = C.p
    @. C.ρ   = abs2(u)                      # ρ = |ψ|^2

    @. C.lnρ = log(C.ρ + p.small)           # lnρ = log(ρ + ε)
    μln = sum(@. C.ρ * C.lnρ)           # media ponderada por ρ

    @. C.Λ   = 2.0 * angle(u)               # ángulo doble
    s = sum(@. C.ρ * sin(C.Λ))              # suma ponderada de senos
    c = sum(@. C.ρ * cos(C.Λ))              # suma ponderada de cosenos
    μΛ = atan(s, c)                          # media circular

    wrap_to_pi!(C.ΔΛ, @. C.Λ - μΛ)          # ΔΛ ∈ [-π, π]

    # du = [ -κ(lnρ - <lnρ>) - i*(ν/2)*ΔΛ ] * u
    @. du = ( -p.κ*(C.lnρ - μln) - complex(T)(0,1)*(p.ν/2)*C.ΔΛ ) * u
    return nothing
end

# -------------------- 8) jvp: producto direccional J*v (sin autodiff, ni jacobiano) ------
# Para f_impl(u) = -(i/ħ) (T u + V.*u), su derivada es lineal: Jv = -(i/ħ) (T v + V.*v)
# Lo implementamos en GPU. OrdinaryDiffEq lo usará para formar Wv = v - γ Jv en GMRES.

# ND
function jvp_schrodinger!(Jv::AbstractArray{Complex{T},N}, v::AbstractArray{Complex{T},N}, C::SchrCache1D{T}, t) where {T, N}
    schrodinger_impl!(Jv, v, C, t)
    return nothing
end



# -------------------- 9) ESTADOS INICIALES (sin indexado escalar; usan dx/dy del cache) ---
# Gauss fundamental del oscilador armónico (puedes reemplazar por tu coherente si quieres)

# Estado coherente 1D (GPU, sin indexado escalar)
# Mantiene exactamente tu forma: exp( -(x- x̄)^2 / (2*Δx)^2 )  (den = (2Δx)^2)
function coherent1D(x::AbstractArray{T,1}, α0::Complex{T}, σ::T, ω::T, t::T) where T
    αt  = α0 * exp(-1im*(ω*t - σ))
    x̄  = sqrt(2/ω) * real(αt)
    p̄  = sqrt(2*ω) * imag(αt)
    Δx  = sqrt(1/(2*ω))
    θ   = -ω*t/2 + (abs2(α0)*sin(2*ω*t - 2σ))/2     # abs2 == |α0|^2
    φ   = exp(1im*θ)
    pref= (ω/π)^(1/4)
    den = (2*Δx)^2                                  # <- así reproducimos tu denominador

    # ψ(x,t) = pref * φ * exp(-((x-x̄)^2)/den) * exp(i p̄ x)
    ψ = @. (pref) * (φ) *
              exp(-((x - x̄)^2)/den) *
              cis(p̄ * x)                           # cis(z)=exp(i z), estable y GPU-friendly
    return Complex{T}.(ψ)                                      # NO normalizamos aquí
end

# Estado entrelazado 2D (GPU): solo el término c1 · ψ_rx(x) · ψ_ry(y)
# Devuelve Ny×Nx (primera dim y), consistente con V, K2, ψk, tmp del caché.
function entangled_ψ(x::AbstractArray{T,1}, y::AbstractArray{T,1},
    α0x::Complex{T}, σx::T, ωx::T,
    α0y::Complex{T}, σy::T, ωy::T,
    t::T, c1::Complex{T}, c2::Complex{T}) where T
    # Coherentes 1D en cada eje (sin normalizar)
    ψrx = coherent1D(x, α0x, σx, ωx, t)    # Nx
    ψry = coherent1D(y, α0y, σy, ωy, t)    # Ny

    Nx = length(x); Ny = length(y)

    # Construye Ny×Nx = ψ_ry(y) * ψ_rx(x) (outer product con broadcasting)
    # reshape(ψry, Ny,1) .* reshape(ψrx,1,Nx) → Ny×Nx
    ψ = @. c1 * reshape(ψry, Ny, 1) * reshape(ψrx, 1, Nx)

    # Si más adelante activas el segundo término (comentado en tu idea):
    # ψlx = coherent1D_gpu(x, α0x, σx+π, ωx, t)
    # ψly = coherent1D_gpu(y, α0y, σy+π, ωy, t)
    # ψ  += @. c2 * reshape(ψly, Ny, 1) * reshape(ψrx, 1, Nx) # o la combinación que quieras

    return Complex{T}.(ψ)                                      # NO normalizamos aquí
end

# -------------------- 10) CONSTRUCCIÓN DEL SPLIT PROBLEM (1D / 2D) -----------------------
# Nota: ODEFunction con keyword :jvp (NO Wmul/Wfact). Esto evita autodiff en GPU.

function build_problem_1d(cfg::Config{T}, to_device, Vfun, ψ0) where T
    C = make_cache_1d(cfg, to_device, Vfun)
    # Estado inicial:
    if isnothing(ψ0)
        ψ = coherent1D(C.x, cfg.α0x, cfg.σx, C.p.ωx, cfg.tspan[1])     # t0
        # Normalización discreta: ∑ |ψ|^2 = 1
        # norm2 = sum(abs2.(ψ))
        # ψ ./= sqrt(norm2)
    else
        ψ = complex(T).(ψ0) |> to_device
    end

    f_impl! = (du,u,p,t) -> schrodinger_impl!(du, u, C, t)
    jvp!    = (Jv,v,u,p,t) -> jvp_schrodinger!(Jv, v, C, t)
    f_expl! = (du,u,p,t) -> decoherence!(du, u, C, t)

    F_impl  = ODEFunction(f_impl!; jvp=jvp!)
    F_expl  = ODEFunction(f_expl!)
    prob    = SplitODEProblem(F_impl, F_expl, ψ, cfg.tspan, C)
    return prob, C
end

function build_problem_2d(cfg::Config{T}, to_device, Vfun, ψ0) where {T}
    C = make_cache_2d(cfg, to_device; Vfun)
    if isnothing(ψ0)
        ψ = entangled_ψ(C.x, C.y,
                            cfg.α0x, cfg.σx, C.p.ωx,
                            cfg.α0y, cfg.σy, C.p.ωy,
                            cfg.tspan[1], cfg.c1, cfg.c2)
        # Normalización discreta 2D: ∑ |ψ|^2 = 1
        # norm2 = sum(abs2.(ψ))
        # ψ ./= sqrt(norm2)
    else
        ψ = complex(T).(ψ0) |> to_device
    end

    f_impl! = (du,u,p,t) -> schrodinger_impl!(du, u, C, t)
    jvp!    = (Jv,v,u,p,t) -> jvp_schrodinger!(Jv, v, C, t)
    f_expl! = (du,u,p,t) -> decoherence!(du, u, C, t)

    F_impl  = ODEFunction(f_impl!; jvp=jvp!)
    F_expl  = ODEFunction(f_expl!)
    prob    = SplitODEProblem(F_impl, F_expl, ψ, cfg.tspan, C)
    return prob, C
end


# -------------------- 12) RESOLVER: KenCarp58 + GMRES (JFNK, sin autodiff) ---------------
function solve_problem(cfg::Config{T}, to_device; Vfun=nothing, ψ0=nothing) where T
    prob, cache = if cfg.dims==1
        if isnothing(Vfun)
            build_problem_1d(cfg, to_device, harmonic_V_1d, ψ0)
        else
            build_problem_1d(cfg, to_device, Vfun, ψ0)
        end
    elseif cfg.dims==2
        if isnothing(Vfun)
            build_problem_2d(cfg, to_device, harmonic_V_2d, ψ0)
        else
            build_problem_2d(cfg, to_device, Vfun, ψ0)
        end
    else
        @error "nothing to do with cfg.dims != (1,2)"
    end

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
function main(T; kwargs...)
    if CUDA.functional()
        to_device(x::AbstractArray) = CuArray(x)
    else
        to_device(x::AbstractArray) = x
    end
    cfg = Config{T}(; kwargs...)
    sol, cache = solve_problem(cfg, to_device)
    @info "Listo. Estados guardados: $(length(sol.t))  |  t_final = $(sol.t[end])"
    return sol, cache
end

function main_1()
    T = Float32
    p = (
    dims = 1,
    Nx = 1024,
    Lx = T(30),
    κ = T(0),
    ν = T(0),
    α0x = Complex{T}(2.5, 1),
    σx = T(0.0),
    ωx = T(1),
    tspan = (T(0.0), T(5.0))
    )

    sol1, cache1 = main(T; p...)

    # 1) Toma x y ψ_sim del resultado, y pásalos a CPU
    x_cpu   = Array(cache1.x)          # cache1 lo devolvió main(...)
    ψ_sim   = Array(sol1.u[end])       # último estado en t = tspan[2]
    ψ_sim_r = real.(ψ_sim)             # parte real (o usa abs.(ψ_sim) si quieres módulo)



    # 3) Construye la analítica (elige tus α0 y σ)
    t_f = sol1.t[end]     # último tiempo del numérico
    ψf   = coherent1D(x_cpu, p.α0x, p.σx, p.ωx, t_f)
    ψf_r = real.(ψf)

    # 4) Grafica
    fig = Figure()
    ax  = Axis(fig[1,1], xlabel="x", ylabel="Re ψ(x)")
    lines!(ax, x_cpu, ψ_sim_r, label="Simulado (GPU)", linestyle=:dash)
    lines!(ax, x_cpu, ψf_r,    label="Analítico (coherente)")
    axislegend(ax)
    display(fig)
end
