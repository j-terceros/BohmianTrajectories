using SummationByPartsOperators: AbstractDerivativeOperator
using Revise
using OrdinaryDiffEq
using Sundials
using SummationByPartsOperators

includet("utils.jl")

function main()
    # initialize config
    T = Float64
    cfg = Config{T}(;
        dims = 1,
        N = 1024,
        Lx = T(30),
        κ = T(-1),
        ν = T(1),
        α0x = Complex{T}(2.5, 0),
        σx = T(0.0),
        ωx = T(1),
        tspan = (T(0.0), T(1.0)),
        dtmax = T(0.05),
        saveat = T(0.01)
    )
    
    # initialize cache
    ch = make_cache_logpsi(cfg)

    # initialize wave function
    ψ_0 = coherent1D(cfg)

    # convert to log psi

    # define problem

    # define algorithm

    # solve problem
end

struct CacheLogPsi{T<:AbstractFloat,N,M,LeftBC,RightBC}
    ψ::AbstractArray{T,N}
    ψ_C::AbstractArray{Complex{T},M}
    prob::AbstractArray{T,M}
    D::AbstractDerivativeOperator{T}
    left_bc::LeftBC
    right_bc::RightBC
end

function make_cache_logpsi(cfg::Config{T}) where T<:AbstractFloat
    make_cache_logpsi(Val(cfg.dims), cfg)
end

function make_cache_logpsi(::Val{1}, cfg::Config{T}) where T<:AbstractFloat
    # Derivative operator
    Dx = derivative_operator(MattssonSvärdNordström2004(); derivative_order=2, accuracy_order=4, 
        xmax=cfg.Lx/2, xmin=-cfg.Lx/2, N = cfg.N, mode=ThreadedMode())
    ψ_C = rand(Complex{T}, cfg.N)
    ψ = convert_psi_to_real(ψ_C)
    prob = prob_psi(ψ_C)

    # other bc not implemented yet
    left_bc = Val(:HomogeneousNeumann)
    right_bc = Val(:HomogeneousNeumann)
    
    return CacheLogPsi{T,2,1,typeof(left_bc),typeof(right_bc)}(ψ, ψ_C, prob, Dx, left_bc, right_bc)
end

# missing cache psi

function schr_harosc_logpsi!(::Val{1}, du, u, p, t)
    ch = p.ch

    # get psi
    convert_gamma_real_to_psi_real!(ch.ψ, u, ch.ψ_C)

    # call schro in psi
    schr_harosc!(Val(1), du, ch.ψ, p, t)

    # divide by psi to go back to gamma
    prob_psi!(ch.prob, ch.ψ)
    r_ψ, i_ψ = eachslice(u, dims=1)

    @inbounds begin
        for i in eachindex(ch.prob)
            if ch.prob[i] > eps(ch.T)
                du[1,i] = (du[1,i]*r_ψ + du[2,i]*i_ψ) / ch.prob[i]
                du[2,i] = (-du[1,i]*i_ψ + du[2,i]*r_ψ) / ch.prob[i]
            else
                du[1,i] = 0
                du[2,i] = 0
            end
        end
    end

    return nothing
end

function schr_harosc!(::Val{1}, du, u, p, t)
    ch = p.ch
    cfg = p.cfg

    r_du, i_du = eachslice(du, dims=1)
    r_ψ, i_ψ = eachslice(u, dims=1)
     
    # harmonic potential
    map!((x,psi)->x^2*psi , i_du, grid(ch.D), r_ψ )
    map!((x,psi)->x^2*psi , r_du, grid(ch.D), i_ψ )

     
    # second derivative of psi
    mul!(r_du, ch.D, i_ψ, -cfg.ħ/(2*cfg.m), cfg.m*cfg.ωx^2/(2*cfg.ħ))
    mul!(i_du, ch.D, r_ψ, cfg.ħ/(2*cfg.m), -cfg.m*cfg.ωx^2/(2*cfg.ħ))

    # boundary correction
    boundary_correction!(Val(1), du, u, p, t)

    return nothing
end

function boundary_correction!(::Val{1}, du, u, p, t)
    ch = p.ch 
    left_bc = ch.left_bc
    right_bc = ch.right_bc
    r_ψ, i_ψ = eachslice(u, dims=1)
    
    # boundary conditions using SATs
    if left_bc == Val(:HomogeneousNeumann)
        du[1,1]  += derivative_left(ch.D, r_ψ, Val(1))  / left_boundary_weight(ch.D)
        du[2,1]  += derivative_left(ch.D, i_ψ, Val(1))  / left_boundary_weight(ch.D)
    else
        throw(ArgumentError("Boundary condition $left_bc not implemented."))
    end

    if right_bc == Val(:HomogeneousNeumann)
        du[1,end] -= derivative_right(ch.D, r_ψ, Val(1)) / right_boundary_weight(ch.D)
        du[2,end] -= derivative_right(ch.D, i_ψ, Val(1)) / right_boundary_weight(ch.D)
    else
        throw(ArgumentError("Boundary condition $right_bc not implemented."))
    end
    return nothing
end


# missing deco eq in log psi
# missing deco eq in psi
