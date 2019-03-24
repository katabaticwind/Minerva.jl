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
const MAXSTEPS = 200


"""
CartPole

	Sutton [].
"""
mutable struct CartPole <: AbstractEnvironment
    state::Vector{Float64}
    action_space
	maxsteps::Int
	rendered
end

CartPole() = CartPole(0.1 * rand(4) .- 0.05, [1, 2], MAXSTEPS, false)

function reset!(env::CartPole)
    env.state = 0.1 * rand(4) .- 0.05
    env.action_space = [1, 2]
	env.maxsteps = MAXSTEPS
	env.rendered && render(env)
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
	env.rendered && render(env)
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

function render(env::CartPole)
	env.rendered = true
	x, xvel, θ, θvel = env.state
	θ = θ + pi / 2
	plot([x, x + 2 * pole_length * cos(θ)],
		 [0.0, 2 * pole_length * sin(θ)],
		  linewidth = 3,
		  legend = false,
		  xlim = (-2.4, 2.4),
		  ylim = (-1.25, 1.25));
	display(scatter!([x], [0.0], color = 4, markersize = 6))
end
