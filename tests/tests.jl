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
end
