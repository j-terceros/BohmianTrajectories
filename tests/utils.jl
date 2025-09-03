using OrdinaryDiffEq
using GLMakie
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
    coherent1D(x, cfg.α0x, cfg.σx, cfg.ωx, t; renorm=renorm) where T
end

function entangled_ψ(x::AbstractArray{T,1}, y::AbstractArray{T,1}, t, cfg::Config{T};
                        renorm = false, entang = false) where T
    entangled_ψ(x, y,
    cfg.α0x, cfg.σx, cfg.ωx,
    cfg.α0y, cfg.σy, cfg.ωy,
    t, cfg.c1, cfg.c2;
    renorm = renorm, entang = enang) where T
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

    # Construye Ny×Nx = ψ_ry(y) * ψ_rx(x) (outer product con broadcasting)
    ψ = c1 .* reshape(ψry, Ny, 1) .* reshape(ψrx, 1, Nx)

    if entang
        ψlx = coherent1D(x, α0x, σx+π, ωx, t)
        ψly = coherent1D(y, α0y, σy+π, ωy, t)
        ψ  += @. c2 * reshape(ψly, Ny, 1) * reshape(ψrx, 1, Nx) # o la combinación que quieras
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
function convert_psi_to_real(ψ::AbstractArray{Complex{T},N}) where {T,N}
    vcat(real(ψ), imag(ψ))
end

function convert_psi_to_complex(ψ::AbstractArray{T,1}) where {T<:AbstractFloat}
    Nx = length(ψ)
    psi_re = @view ψ[1:Nx]
    psi_im = @view ψ[Nx+1:end]
    return psi_re + im*psi_im
end

function convert_psi_to_complex(ψ::AbstractArray{T,2}) where {T<:AbstractFloat}
    Nx = size(ψ,1)
    psi_re = @view ψ[1:Nx,:]
    psi_im = @view ψ[Nx+1:end,:]
    return psi_re + im*psi_im
end

function prob_psi(ψ::AbstractArray{T,1}) where{T<:AbstractFloat}
    Nx = length(ψ)
    psi_re = @view ψ[1:Nx]
    psi_im = @view ψ[Nx+1:end]

    return psi_re.^2 .+ psi_im.^2
end

function prob_psi(ψ::AbstractArray{T,2}) where{T<:AbstractFloat}
    Nx = size(ψ,1)
    psi_re = @view ψ[1:Nx, :]
    psi_im = @view ψ[Nx+1:end, :]

    return psi_re.^2 .+ psi_im.^2
end

function prob_psi(ψ::AbstractArray{Complex{T},N}) where where{T,N}
    prob = abs2.(ψ)
end

function width(ψ::AbstractArray{T,1}, x) where{T<:AbstractFloat}
    prob = prob_psi(ψ)

    Z = sum(prob)

    mean_x = sum(prob .* x) / Z

    mean_x2 = sum(prob .* x.^2) / Z

    return sqrt(mean_x2 - mean_x^2)
end


#################### 
# plotting results #
#################### 
function plot_1D_comparision(sol_psi, x_cpu)
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
