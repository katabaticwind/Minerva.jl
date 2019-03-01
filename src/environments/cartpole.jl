# Based on Reinforce.jl (ported from OpenAI)

const gravity = 9.8
const mass_cart = 1.0
const mass_pole = 0.1
const total_mass = mass_cart + mass_pole
const pole_length = 0.5  # half the pole's length
const mass_pole_length = mass_pole * pole_length
const force_mag = 10.0
const τ = 0.02  # seconds between state updates
const θ_threshold = 24π / 360  # angle at which to fail the episode
const x_threshold = 2.4


"""
CartPole

	Sutton [].
"""
mutable struct CartPole <: AbstractEnvironment
    state::Vector{Float64}
    action_space
	maxsteps::Int
end

CartPole() = CartPole(0.1 * rand(4) .- 0.05, [1, 2], 500)

function reset!(env::CartPole)
    env.state = 0.1 * rand(4) .- 0.05
    env.action_space = [1, 2]
	env.maxsteps = 500
    return env.state, false
end

function step!(env::CartPole, action)
    @assert(action in env.action_space)
	x, xvel, θ, θvel = env.state
    force = (action == 1 ? -1 : 1) * force_mag
    tmp = (force + mass_pole_length * sin(θ) * (θvel^2)) / total_mass
    θacc = (gravity * sin(θ) - tmp * cos(θ)) / (pole_length * (4/3 - mass_pole * (cos(θ)^2) / total_mass))
    xacc = tmp - mass_pole_length * θacc * cos(θ) / total_mass
	env.state = [x + τ * xvel, xvel + τ * xacc, θ + τ * θvel, θ + τ * θacc]
	done = isdone(env)
	r = done ? 0.0 : 1.0
    info = nothing
    if !done
        env.action_space = available_actions(env)
    end
    return env.state, r, done, info
end

function available_actions(env::CartPole)
    return [1, 2]
end

function isdone(env::CartPole)
  x, xvel, θ, θvel = env.state
  !(-x_threshold <= x <= x_threshold &&
    -θ_threshold <= θ <= θ_threshold)
end
