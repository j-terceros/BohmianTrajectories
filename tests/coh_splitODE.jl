using DifferentialEquations
using FFTW
using GLMakie

Nx = 1024
Lx = 25.0
Δx = Lx / Nx
x = collect(range(-Lx/2, stop=Lx/2 - Δx, length=Nx))
kx  = 2π .* fftfreq(Nx, 1/Δx)


α0x = 5/2
σx  = 0.0
ωx  = 1.0
const ħ = 1.0
const m = 1.0

tspan = (0.0, 10.0)

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

function schrodinger(dψ, ψ, p, t)
    dψ .= fV!(dψ, ψ, p, t) + fT!(dψ, ψ, p, t)
end

ψ0 = coherent1D(x, α0x, σx, ωx, 0.0)

prob = SplitODEProblem(fT!, fV!, ψ0, tspan)

sol = solve(prob, KenCarp47(autodiff = AutoFiniteDiff()), reltol=1e-6, abstol=1e-6; saveat = 0.1)

t_vals = tspan[1]:0.1:tspan[2]
ψf = coherent1D(x, α0x, σx, ωx, tspan[2])


fig = Figure()
ax = Axis(fig[1, 1], xlabel="X", ylabel="Real Part of the Wave Function - Final State")
lines!(ax, x, real.(sol.u[end]), label="Simulated", color=:blue, linestyle=:dash)
lines!(ax, x, real.(ψf), label="Analytical", color=:red)
axislegend(ax)
display(fig)
