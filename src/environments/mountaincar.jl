# Based on Reinforce.jl (ported from OpenAI)

const min_position = -1.2
const max_position = 0.6
const max_speed = 0.07
const goal_position = 0.5
const min_start = -0.6
const max_start = 0.4
const car_width = 0.05
const car_height = car_width / 2.0
const clearance = 0.2 * car_height
const flag_height = 0.05


"""
MountainCar
"""
mutable struct MountainCar <: AbstractEnvironment
		state::Vector{Float64}  # [position, velocity]
		action_space
		maxsteps::Int
end

MountainCar() = MountainCar([0.0, 0.0], [1, 2, 3], 500)

function reset!(env::MountainCar)
    env.state = [0.0, 0.0]
    env.action_space = [1, 2, 3]
    return env.state, false
end

function step!(env::MountainCar, action)
    @assert(action in env.action_space)
		position, velocity = env.state
	  velocity += (action - 2) * 0.001 + cos(3 * position) * (-0.0025)
	  velocity = clamp(velocity, -max_speed, max_speed)
	  position += velocity
	  if position <= min_position && velocity < 0
	    velocity = 0
	  end
	  position = clamp(position, min_position, max_position)
	  env.state = [position, velocity]
		done = isdone(env)
	  r = -1
		info = nothing
		if !done
			env.action_space = available_actions(env)
		end
		return env.state, r, done, info
end

function available_actions(env::MountainCar)
    return [1, 2, 3]
end

function isdone(env::MountainCar)
		env.state[1] >= goal_position
end
