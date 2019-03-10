abstract type AbstractScheduler end

struct StepDecay <: AbstractScheduler
    xmax
    xmin
    Δ
    T
end

(d::StepDecay)(t) = max(d.xmin, d.xmax - div(t, d.T) * d.Δ)


d = StepDecay(1.0, 0.1, 0.1, 5)
plot(1:100, d.(1:100))


struct ExponentialDecay <: AbstractScheduler
    xmax
    xmin
    ρ
end

(d::ExponentialDecay)(t) = max(d.xmin, d.xmax * d.ρ^t)

d = ExponentialDecay(1.0, 0.1, 0.95)
plot(1:100, d.(1:100))
