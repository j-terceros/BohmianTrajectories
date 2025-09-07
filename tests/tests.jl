using SafeTestsets, Test
using Revise
includet("utils.jl")

@testset "utils" begin
    @testset "psi complex to real and back" begin
        T=Float64
        n = 4
        m = 4

        # 1D
        ψ = rand(Complex{T}, n)
        psi = convert_psi_to_real(ψ)
        @test isapprox(psi[1,:], real(ψ))
        @test isapprox(psi[2,:], imag(ψ))
        ψ_ = convert_psi_to_complex(psi)
        @test isapprox(ψ, ψ_)


        # 2D
        ψ = rand(Complex{T}, n,m)
        psi = convert_psi_to_real(ψ)
        @test isapprox(psi[1,:,:], real(ψ))
        @test isapprox(psi[2,:,:], imag(ψ))
        ψ_ = convert_psi_to_complex(psi)
        @test isapprox(ψ, ψ_)
    end

    @testset "psi probability" begin
        T=Float64
        n = 4
        m = 4

        # 1D
        ψ = rand(Complex{T}, n)
        prob = prob_psi(ψ)
        @test isapprox(prob, real.(ψ).^2 .+ imag.(ψ).^2)
        psi = convert_psi_to_real(ψ)
        prob_ = prob_psi(psi)
        @test isapprox(prob_, real.(ψ).^2 .+ imag(ψ).^2)

        # 2D
        ψ = rand(Complex{T}, n,m)
        prob = prob_psi(ψ)
        @test isapprox(prob, real.(ψ).^2 .+ imag.(ψ).^2)
        psi = convert_psi_to_real(ψ)
        prob_ = prob_psi(psi)
        @test isapprox(prob_, real.(ψ).^2 .+ imag.(ψ).^2)
    end

    @testset "gamma psi convert" begin
        T = Float64
        n = 4
        m= 4

        # 1D
        ψ_C = rand(Complex{T}, n)
        ψ = convert_psi_to_real(ψ_C)
        γ = similar(ψ)
        ψ_ = similar(ψ)
        aux = similar(ψ_C)
        convert_psi_real_to_gamma_real!(γ, ψ, aux)
        @test isapprox(γ[1,:], real.(log.(ψ_C)))
        @test isapprox(γ[2,:], imag.(log.(ψ_C)))

        convert_gamma_real_to_psi_real!(ψ_, γ, aux)
        @test isapprox(ψ_, ψ)
    end

    @testset "ssfm coherent 1D" begin
        T = Float64
        cfg = Config{T}(;dtmax=T(0.001), tspan=(T(0), T(1)))

        # make grid
        x = LinRange(-cfg.Lx/2*(1 - 1/cfg.N), cfg.Lx/2*(1 - 1/cfg.N), cfg.N); # move half-step

        # initial condition
        ψ_0 = convert_psi_to_real(coherent1D(x, cfg, cfg.tspan[1]) )

        # evolve using ssfm
        ssfm_harosc!(ψ_0, cfg)

        # final analytic value
        ψ_f = convert_psi_to_real(coherent1D(x, cfg, cfg.tspan[2]) )
        @show norm(ψ_0 - ψ_f)
        @show maximum(abs2.(ψ_0 - ψ_f))

        @test isapprox(ψ_0, ψ_f, atol=cfg.dtmax^2, rtol=cfg.dtmax^2*10)
    end
end

includet("schr_dec.jl")

@testset "schr_dec schr" begin
    T = Float64
    cfg = Config{T}(; tspan=(T(0), T(1)))

    p = (
        ch = make_cache_logpsi(cfg),
        cfg = cfg)
     
    # initial condition
    ψ_0 = convert_psi_to_real(coherent1D(grid(p.ch.D), cfg, cfg.tspan[1]) )

    function f!(du, u, p, t)
        schr_harosc!(Val(1), du, u, p, t)
    end


    prob = ODEProblem(f!, ψ_0, cfg.tspan, p)
    common =(
        reltol = 1e-7,
        abstol = 1e-10,
        dtmax = 0.001,
        maxiters=Int(1e6),
    )

    alg = CVODE_BDF(linear_solver=:GMRES, stability_limit_detect=true)
    sol = solve(prob, alg; common...)

    # final analytic value
    ψ_f = convert_psi_to_real(coherent1D(grid(p.ch.D), cfg, cfg.tspan[2]) )

    @show norm(sol(cfg.tspan[2]) - ψ_f)
    @show maximum(abs2.(sol(cfg.tspan[2]) - ψ_f))

    @test isapprox(sol(cfg.tspan[2]), ψ_f, atol=common.dtmax^2, rtol=common.dtmax^2*10)
end
