using DifferentialEquations
using FFTW
using GLMakie

const ħ = 1.0
const m = 1.0

function main()
    Nx = 1024
    Lx = 30.0
    dx = Lx / Nx
    x = collect(range(-Lx/2, stop=Lx/2 - dx, length=Nx));
    kx  = 2π .* fftfreq(Nx, 1/dx);


    α0x = 0.0
    σx  = 0.0
    ωx  = 1.0
    κ   = -0.01
    ν   = 0.01
    small = eps()

    p = (ħ, m, κ, ν, ωx);

    tspan = (0.0, 1.0);

    # Reusable FFT plans and buffers
    # ψbuf  = zeros(ComplexF64, Nx)
    ψbuf  = zeros(ComplexF64, 2*Nx);
    planF = plan_fft(ψbuf[1:Nx]);
    planB = plan_ifft(ψbuf[1:Nx]);

    Vx = 0.5 * m * ωx^2 .* (x.^2);
    Tx = 0.5 * ħ^2 / m .* (kx.^2 );

    function coherent1D(x, α0, σ, ω, t)
        αt = α0 * exp(-1im*(ω*t - σ))
        x̄ = sqrt(2/ω)*real(αt)
        p̄ = sqrt(2*ω)*imag(αt)
        Δx = sqrt(1/(2ω))
        θ = -ω*t/2 + (abs(α0)^2*sin(2*ω*t - 2σ))/2
        φ = exp(1im*θ)
        pref = (ω/π)^(1/4)
        psi = pref .* φ .* exp.(-((x.-x̄).^2)/(2*Δx)^2) .* exp.(1im*p̄.*x)
        return vcat(real(psi), imag(psi))
    end

    function fT!(dψ, ψ, p, t)
        psi_re = @view ψ[1:Nx]
        psi_im = @view ψ[Nx+1:end]
        ψbuf[1:Nx] .=  Tx .* (planF * psi_im) ./ ħ
        ψbuf[Nx+1:end] .=  -Tx .* (planF * psi_re) ./ ħ 
        dψ[1:Nx] .= real.(planB * ψbuf[1:Nx])      
        dψ[Nx+1:end] .= real.(planB * ψbuf[Nx+1:end])       
    end

    function fV!(dψ, ψ, p, t)
        psi_re = @view ψ[1:Nx]
        psi_im = @view ψ[Nx+1:end]
        dψ[1:Nx] .=  Vx .* (psi_im) ./ ħ
        dψ[Nx+1:end] .=  -Vx .* (psi_re) ./ ħ 
    end

    function schrodinger!(dψ, ψ, p, t)
        fT!(dψ, ψ, p, t)
        fV!(dψ, ψ, p, t)
    end

    function decoherence!(dψ, ψ, p, t)
        psi_re = @view ψ[1:Nx]
        psi_im = @view ψ[Nx+1:end]

        ρ = @. psi_re^2 + psi_im^2 

        Z = sum(ρ)

        lnρ = log.(ρ .+ small)
        mean_lnρ = sum(ρ .* lnρ) / Z

        Λ = atan.(psi_im, psi_re) .* 2
        mean_Λ = sum(ρ .* Λ) / Z

        @. dψ[1:Nx] =  -κ * (lnρ - mean_lnρ) * psi_re - (ν/2) * (Λ - mean_Λ) * psi_re
        @. dψ[Nx+1:end] =  -κ * (lnρ - mean_lnρ) * psi_im - (ν/2) * (Λ - mean_Λ) * psi_im
    end

    function width(ψ, x, dx)
        psi_re = @view ψ[1:Nx]
        psi_im = @view ψ[Nx+1:end]

        prob = @. psi_re^2 + psi_im^2

        Z = sum(prob)

        mean_x = sum(prob .* x) / Z

        mean_x2 = sum(prob .* x.^2) / Z

        return sqrt(mean_x2 - mean_x^2)
    end

    function rho(ψ, x, dx)

        psi_re = @view ψ[1:Nx]
        psi_im = @view ψ[Nx+1:end]

        prob = @. psi_re^2 + psi_im^2

        return sum(prob)
    end

    function diff_width(du, u, p, t)
        δ, dδ = u
        ħ, m, κ, ν, ωx = p
        du[1] = dδ
        du[2] = (2*κ - ν)*dδ + (ν*κ - κ^2)*δ + (ħ^2) / (4*m^2*δ^3) - δ*ωx^2
    end

    ψ0 = coherent1D(x, α0x, σx, ωx, 0.0);

    prob = SplitODEProblem(decoherence!, schrodinger!, ψ0, tspan);

    sol = solve(prob, KenCarp47(), reltol=1e-13, abstol=1e-13; saveat = 0.001);

    t_vals = tspan[1]:0.1:tspan[2];

    ψf = coherent1D(x, α0x, σx, ωx, tspan[2]);

    δ0 = [sqrt(1/(2ωx)), κ*sqrt(1/(2ωx))];

    δf = width(ψf, x, dx);

    prob_δ = ODEProblem(diff_width, δ0, tspan, p);

    sol_δ = solve(prob_δ, KenCarp47(), reltol=1e-12, abstol=1e-12; saveat = 0.001);

    δ_dδ = reduce(vcat, [u' for u in sol_δ.u]);

    t_steps = length(sol_δ.t);

    δ_all = zeros(t_steps);

    for n in 1:t_steps
        δ_all[n] = width(sol.u[n], x, dx)
    end

    function ploting_result()
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel="X", ylabel="Real Part of the Wave Function - Final State (S-ODE)")
        lines!(ax, x, sol.u[end][1:Nx].^2 + sol.u[end][Nx+1:end].^2, label="Decoherence", color=:blue, linestyle=:dash)
        lines!(ax, x, ψf[1:Nx].^2 + ψf[Nx+1:end].^2 , label="No-Decoherence", color=:red)
        axislegend(ax)

        ax1 = Axis(fig[1, 2], xlabel="Time", ylabel="Width")
        lines!(ax1, sol_δ.t, δ_all, label="Simulated", color=:green, linestyle=:dash)
        lines!(ax1, sol_δ.t, δ_dδ[:,1], label="Analytical", color=:red)
        axislegend(ax1)
        return fig
    end
    fig = ploting_result();
    return display(fig)
end
