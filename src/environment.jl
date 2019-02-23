"""
Episode

    A container to keep track of environment feedback during an episode.
"""
mutable struct Episode
    total_reward
    niter
end

Episode() = Episode(0.0, 0)


"""
Environment

    Abstract stateful environment.

    Keeps track of the environment state and implements update eqn. s', r = ρ(s, a).
"""
abstract type AbstractEnvironment end
function reset!(env::AbstractEnvironment) end
function step!(env::AbstractEnvironment, action) end
function available_actions(env::AbstractEnvironment) end

mutable struct BasicEnvironment <: AbstractEnvironment
    state
    action_space
end

BasicEnvironment() = BasicEnvironment([0.0], [1, 2])

function reset!(env::BasicEnvironment)
    env.state = [0.0]
    env.action_space = available_actions(env)
    return env.state, false
end

function step!(env::BasicEnvironment, action)
    @assert(action in env.action_space)
    env.state[1] += randn()
    r = randn()
    done = abs(env.state[1]) > 5.0
    info = nothing
    if !done
        env.action_space = available_actions(env)
    end
    return env.state, r, done, info
end

function available_actions(env::BasicEnvironment)
    return [1, 2]
end


# """
# Policy
#
#     Chooses actions based on the current environment's state.
# """
# abstract type AbstractPolicy end
# function action(π::AbstractPolicy, env::AbstractEnvironment) end
#
# struct RandomPolicy <: AbstractPolicy end
#
# function action(π::RandomPolicy, env::AbstractEnvironment)
#     return rand([1, 2])
# end


# """
# run_episode()
#
#     Run through a complete episode of training.
#
#     At each timestep, choose and action based on the environment's current state, and perform an environment update step using the chosen action. Repeat until reaching a terminal state.
# """
# function run_episode(env::E, π::P) where {E <: AbstractEnvironment, P <: AbstractPolicy}
#     ep = Episode()
#     s, done = reset!(env)
#     while !done
#       # render(env)
#       a = action(π, env)
#       s, r, done, info = step!(env, a)
#       println("s = $s, a = $a, r = $r, done = $done")
#       ep.total_reward += r
#       ep.niter += 1
#     end
#     println("Episode finished after $(ep.niter) timesteps")
#     return ep
# end
