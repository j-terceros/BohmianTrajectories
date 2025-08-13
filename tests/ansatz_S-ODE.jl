using DifferentialEquations
using FFTW
using GLMakie

Nx = 1024
Lx = 30.0
dx = Lx / Nx
x = collect(range(-Lx/2, stop=Lx/2 - dx, length=Nx))
kx  = 2π .* fftfreq(Nx, 1/dx)


α0x = 5/2
σx  = 0.0
ωx  = 1.0
κ   = -0.1
ν   = 0.1
const ħ = 1.0
const m = 1.0

p = (ħ, m, κ, ν, ωx)

tspan = (0.0, 0.5)

# Reusable FFT plans and buffers
ψbuf  = zeros(ComplexF64, Nx)
planF = plan_fft(ψbuf)
planB = plan_ifft(ψbuf)

Vx = 0.5 * m * ωx^2 .* (x.^2)
Tx = 0.5 * ħ^2 / m .* (kx.^2 )

function coherent1D(x, α0, σ, ω, t)
    αt = α0 * exp(-1im*(ω*t - σ))
    x̄ = sqrt(2/ω)*real(αt)
    p̄ = sqrt(2*ω)*imag(αt)
    Δx = sqrt(1/(2ω))
    θ = -ω*t/2 + (abs(α0)^2*sin(2*ω*t - 2σ))/2
    φ = exp(1im*θ)
    pref = (ω/π)^(1/4)
    return pref .* φ .* exp.(-((x.-x̄).^2)/(2*Δx)^2) .* exp.(1im*p̄.*x)
end

function fT!(dψ, ψ, p, t)
    ψbuf .= planF * ψ
    @. ψbuf = -im/ħ * Tx * ψbuf
    dψ .= planB * ψbuf       
end

function fV!(dψ, ψ, p, t)
    @. dψ = -im/ħ * Vx * ψ
end

function schrodinger!(dψ, ψ, p, t)
    # Parte cinética en k: dψ_T = -(i/ħ) * T(k) * ψ
    ψbuf .= planF * ψ
    @. ψbuf = -im/ħ * Tx * ψbuf
    dψ .= planB * ψbuf
    # Parte potencial en x: dψ_V = -(i/ħ) V(x) ψ
    @. dψ += -im/ħ * Vx * ψ
end

function decoherence!(dψ, ψ, p, t)
    ρ = @. abs2(ψ) + 1e-30

    Z = sum(ρ) * dx

    lnρ = log.(ρ)
    mean_lnρ = sum(ρ .* lnρ) * dx / Z

    # Ln(ψ/ψ*) de forma estable: log(ψ) - log(conj(ψ))
    Λ = log.(ψ .+ 1e-30) .- log.(conj.(ψ) .+ 1e-30)
    mean_Λ = sum(ρ .* Λ) * dx / Z

    # Wκ = @. -κ * (lnρ - mean_lnρ)
    # Wν = @. -(ν/2) * (Λ  - mean_Λ)

    @. dψ =  -κ * (lnρ - mean_lnρ) * ψ - (ν/2) * (Λ - mean_Λ) * ψ

end

function width(ψ, x, dx)
    prob = abs2.(ψ)

    Z = sum(prob) * dx

    mean_x = sum(prob .* x) * dx / Z

    mean_x2 = sum(prob .* x.^2) * dx / Z

    return sqrt(mean_x2 - mean_x^2)
end

function diff_width(du, u, p, t)
    δ, dδ = u
    ħ, m, κ, ν, ωx = p
    du[1] = dδ
    du[2] = (2*κ - ν)*dδ + (ν*κ - κ^2)*δ + (ħ^2) / (4*m^2*δ^3) - δ*ωx^2
end

ψ0 = coherent1D(x, α0x, σx, ωx, 0.0)

prob = SplitODEProblem(decoherence!, schrodinger!, ψ0, tspan)

sol = solve(prob, KenCarp47(autodiff = AutoFiniteDiff()), reltol=1e-10, abstol=1e-10; saveat = 0.001)

t_vals = tspan[1]:0.1:tspan[2]

ψf = coherent1D(x, α0x, σx, ωx, tspan[2])

δ0 = [sqrt(1/(2ωx)), 0.0]

δf = width(ψf, x, dx)

prob_δ = ODEProblem(diff_width, δ0, tspan, p)

sol_δ = solve(prob_δ, Tsit5(), reltol=1e-12, abstol=1e-12; saveat = 0.001)

δ_dδ = reduce(vcat, [u' for u in sol_δ.u])

t_steps = length(sol_δ.t)

δ_all = zeros(t_steps)

for n in 1:t_steps
    δ_all[n] = width(sol.u[n], x, dx)
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel="X", ylabel="Real Part of the Wave Function - Final State (S-ODE)")
lines!(ax, x, real.(sol.u[end]), label="Decoherence", color=:blue, linestyle=:dash)
lines!(ax, x, real.(ψf), label="No-Decoherence", color=:red)
axislegend(ax)
display(fig)

ax1 = Axis(fig[1, 2], xlabel="Time", ylabel="Width")
lines!(ax1, sol_δ.t, δ_all, label="Simulated", color=:green, linestyle=:dash)
lines!(ax1, sol_δ.t, δ_dδ[:, 1], label="Analytical", color=:red)
axislegend(ax1)
display(fig)