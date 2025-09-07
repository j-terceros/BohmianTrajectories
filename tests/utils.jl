# using OrdinaryDiffEq
# using GLMakie
using FFTW
using LinearAlgebra

############################
# configuration parameters #
############################
Base.@kwdef struct Config{T<:AbstractFloat}
    dims::Int                         = 1
    N::Int                           = 1024
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
    reltol::T                   = sqrt(eps(T))
    abstol::T                   = sqrt(eps(T))/100
    saveat::T                   = 0.1
    dtmax::T                    = 0.05 
    # for progress bar
    progress::Bool              = true
    progress_steps::Int         = 100
    # Parámetros del estado coherente y entrelazado
    α0x::Complex{T}                   = 2.5 + 0.0im    # amplitud coherente eje x
    σx::T                       = 0.0            # fase inicial eje x
    α0y::Complex{T}                   = 2.5 + 0.0im    # amplitud coherente eje y (2D)
    σy::T                       = 0.0            # fase inicial eje y (2D)
    c1::Complex{T}                    = 1.0 + 0.0im    # coeficiente del término activo en 2D
    c2::Complex{T}                    = 0.0 + 0.0im    # (reservado) segundo término si luego lo activas
end

######################
# Initial conditions #
######################
function coherent1D(x::AbstractArray{T,1}, α0::Complex{T}, σ::T, ω::T, t::T; renorm=false) where T
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
    if renorm
        norm2 = sum(abs2.(ψ))
        ψ ./= sqrt(norm2)
    end
    return Complex{T}.(ψ)               
end

function coherent1D(x::AbstractArray{T,1}, cfg::Config{T}, t; renorm=false) where T
    coherent1D(x, cfg.α0x, cfg.σx, cfg.ωx, t; renorm=renorm)
end

function entangled_ψ(x::AbstractArray{T,1}, y::AbstractArray{T,1}, t, cfg::Config{T};
                        renorm = false, entang = false) where T
    entangled_ψ(x, y,
    cfg.α0x, cfg.σx, cfg.ωx,
    cfg.α0y, cfg.σy, cfg.ωy,
    t, cfg.c1, cfg.c2;
    renorm = renorm, entang = entang)
end

function entangled_ψ(x::AbstractArray{T,1}, y::AbstractArray{T,1},
    α0x::Complex{T}, σx::T, ωx::T,
    α0y::Complex{T}, σy::T, ωy::T,
    t::T, c1::Complex{T}, c2::Complex{T};
    renorm = false, entang = false) where T
    # Coherentes 1D en cada eje (sin normalizar)
    ψrx = coherent1D(x, α0x, σx, ωx, t)    # Nx
    ψry = coherent1D(y, α0y, σy, ωy, t)    # Ny

    Nx = length(x); Ny = length(y)

    # Construye Nx×Ny = ψ_rx(x) * ψ_ry(y) (outer product con broadcasting)
    # kron(Ψrx, ψry) might be better
    ψ = c1 .* reshape(ψrx, Nx, 1) .* reshape(ψry, 1, Ny)

    if entang
        ψlx = coherent1D(x, α0x, σx+π, ωx, t)
        ψly = coherent1D(y, α0y, σy+π, ωy, t)
        ψ  += @. c2 * reshape(ψlx, Nx, 1) * reshape(ψly, 1, Ny) # o la combinación que quieras
    end
    
    if renorm
        norm2 = sum(abs2.(ψ))
        ψ ./= sqrt(norm2)
    end

    return Complex{T}.(ψ)   
end

#########################
# known delta evolution #
#########################
function reference_delta_1d(cfg::Config{T}) where T
    δ0 = [sqrt(1/(2*cfg.ωx)), cfg.κ*sqrt(1/(2*cfg.ωx))];

    prob_δ = ODEProblem(diff_width!, δ0, cfg.tspan, cfg);

    sol_δ = solve(prob_δ, reltol=1e-12, abstol=1e-12, saveat = cfg.saveat);

    return sol_δ
end

function diff_width!(du, u, p, t)
    δ, dδ = u
    du[1] = dδ
    du[2] = (2*p.κ - p.ν)*dδ + (p.ν*p.κ - p.κ^2)*δ + (p.ħ^2) / (4*p.m^2*δ^3) - δ*p.ωx^2
end

##################
# utils with psi #
##################

function convert_psi_to_real!(psi::AbstractArray{T,N}, ψ::AbstractArray{Complex{T},M}) where {T<:AbstractFloat, N, M}
    @assert N == M+1
    @assert size(psi)[2:end] == size(ψ) "wrong sizes on psi and ψ"  
    @assert size(psi)[1] == 2
    r_arr, i_arr = eachslice(psi, dims=1)
    map!(real, r_arr, ψ)
    map!(imag, i_arr, ψ)
    return nothing
end

function convert_psi_to_real(ψ::AbstractArray{Complex{T},M}) where {T<:AbstractFloat, M}
    psi = zeros(T, 2, size(ψ)...)
    convert_psi_to_real!(psi, ψ)
    return psi
end

function convert_psi_to_complex!(psi::AbstractArray{Complex{T},M}, ψ::AbstractArray{T,N}) where {T<:AbstractFloat, N, M}
    @assert N == M+1
    @assert size(ψ)[2:end] == size(psi) "wrong sizes on psi and ψ"  
    @assert size(ψ)[1] == 2
    r_arr, i_arr = eachslice(ψ, dims=1)
    map!((x,y)->x+im*y, psi, r_arr, i_arr)
    return nothing
end

function convert_psi_to_complex(ψ::AbstractArray{T,N}) where {T<:AbstractFloat, N}
    n = size(ψ)[2:end]
    psi = zeros(Complex{T}, n...)
    convert_psi_to_complex!(psi, ψ)
    return psi
end

function prob_psi!(prob::AbstractArray{T,M}, ψ::AbstractArray{T,N}) where{T<:AbstractFloat, N, M}
    @assert N == M+1
    @assert size(ψ)[2:end] == size(prob) "wrong sizes on psi and ψ"  
    @assert size(ψ)[1] == 2
    r_arr, i_arr = eachslice(ψ, dims=1)
    map!((x,y)-> x^2 +y^2, prob, r_arr, i_arr)
    return nothing
end

function prob_psi(ψ::AbstractArray{T,N}) where{T<:AbstractFloat, N}
    n = size(ψ)[2:end]
    rho = zeros(T,n...)
    prob_psi!(rho, ψ)
    return rho
end

function prob_psi!(prob::AbstractArray{T,N}, ψ::AbstractArray{Complex{T},N}) where{T<:AbstractFloat,N}
    @assert size(prob) == size(ψ) "wrong size on prob and ψ"
    map!(abs2, prob, ψ)
end

function prob_psi(ψ::AbstractArray{Complex{T},N}) where {T<:AbstractFloat,N}
    prob = zeros(T, size(ψ)...)
    prob_psi!(prob, ψ)
    return prob
end

function width(ψ::AbstractArray{T,1}, x) where{T}
    prob = prob_psi(ψ)

    Z = sum(prob)

    mean_x = sum(prob .* x) / Z

    mean_x2 = sum(prob .* x.^2) / Z

    return sqrt(mean_x2 - mean_x^2)
end

################
# utils with γ #
################

function convert_gamma_real_to_psi_real!(ψ::AbstractArray{T,N}, γ::AbstractArray{T,N}, aux::AbstractArray{Complex{T},M}) where {T<:AbstractFloat, N, M}
    @assert size(γ)[1] == 2
    @assert size(γ) == size(ψ)
    @assert size(aux) == size(ψ)[2:end]
    r_gamma, i_gamma = eachslice(γ, dims=1)
    map!((x,y)->exp(x+im*y), aux, r_gamma, i_gamma) 
    convert_psi_to_real!(ψ, aux)
    return nothing
end

function convert_psi_real_to_gamma_real!(γ::AbstractArray{T,N}, ψ::AbstractArray{T,N}, aux::AbstractArray{Complex{T},M}) where {T<:AbstractFloat, N, M}
    @assert size(γ)[1] == 2
    @assert size(γ) == size(ψ)
    @assert size(aux) == size(ψ)[2:end]
    r_psi, i_psi = eachslice(ψ, dims=1)
    map!((x,y)->log(x+im*y), aux, r_psi, i_psi) 
    convert_psi_to_real!(γ, aux)
    return nothing
end

function ssfm_harosc!(ψ::AbstractArray{T,N}, cfg::Config{T}) where {T<:AbstractFloat,N}
    FFTW.set_num_threads(Threads.nthreads())

    # make grid
    x = LinRange(-cfg.Lx/2*(1 - 1/cfg.N), cfg.Lx/2*(1 - 1/cfg.N), cfg.N); # move half-step

    # make fft plans
    du = similar(ψ)

    plan_f  = FFTW.plan_dct(du, 2; flags=FFTW.MEASURE)
    k = π .* (collect(0:cfg.N-1)) / cfg.Lx;
    
    dt  = cfg.dtmax
    t0, tf = cfg.tspan
    steps  = Int(cld(tf - t0, dt))    # ceil division
    dt_eff = (tf - t0) / steps        # adjust dt to hit tf exactly
    
    # Kinetic entire-step: exp(-i (ħ/(2m)) k^2 dt)
    r_t = @. cos((cfg.ħ/(2cfg.m)) * k^2 * dt_eff)
    i_t = @. -sin((cfg.ħ/(2cfg.m)) * k^2 * dt_eff)

    # Potential half-step: exp(-i (ω^2*m/(2ħ)) x^2 dt/2)
    r_v = @. cos( x^2*cfg.ωx^2*cfg.m/(2*cfg.ħ) * dt_eff/2 )
    i_v = @. -sin( x^2*cfg.ωx^2*cfg.m/(2*cfg.ħ) * dt_eff/2 )


    for i in 1:steps 
        r_ψ, i_ψ = eachslice(ψ, dims=1)
        r_du, i_du = eachslice(du, dims=1) 

        # potential half-step
        map!((x,y,a,b)-> a*x - b*y, r_du, r_ψ, i_ψ, r_v, i_v)  
        map!((x,y,a,b)-> b*x + a*y, i_du, r_ψ, i_ψ, r_v, i_v)  

        # kinetic entire-step
        mul!(ψ, plan_f, du)
        map!((x,y,a,b)-> a*x - b*y, r_du, r_ψ, i_ψ, r_t, i_t)  
        map!((x,y,a,b)-> b*x + a*y, i_du, r_ψ, i_ψ, r_t, i_t)  
        ldiv!(ψ, plan_f, du)

        # potential half-step
        map!((x,y,a,b)-> a*x - b*y, r_du, r_ψ, i_ψ, r_v, i_v)  
        map!((x,y,a,b)-> b*x + a*y, i_du, r_ψ, i_ψ, r_v, i_v)  

        # move all to ψ
        map!(identity, r_ψ, r_du)
        map!(identity, i_ψ, i_du)
    end

    return nothing
end

#################### 
# plotting results #
#################### 
function plot_1D_comparision(sol_psi, x_cpu, cfg::Config{T}) where {T}
    sol_δ = reference_delta_1d(cfg)
    δ_dδ = reduce(vcat, [u' for u in sol_δ.u]);
    t_steps = length(sol_δ.t);
    δ_all = zeros(t_steps);
    for i in 1:t_steps
        δ_all[n] = width(Array(sol_psi(t)), x_cpu)
    end

    # 4) Grafica
    fig = Figure()

    ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="Width")
    lines!(ax1, sol_δ.t, δ_all, label="Simulated", color=:green, linestyle=:dash)
    lines!(ax1, sol_δ.t, δ_dδ[:,1], label="Analytical", color=:red)
    axislegend(ax1)

    display(fig)
    return fig
end
