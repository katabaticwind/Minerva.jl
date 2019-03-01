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


"""
RandomWalk

    Random walk with random rewards and meaningless actions.
"""
mutable struct RandomWalk <: AbstractEnvironment
    state
    action_space
end

RandomWalk() = RandomWalk([0.0], [1, 2])

function reset!(env::RandomWalk)
    env.state = [0.0]
    env.action_space = [1, 2]
    return env.state, false  # state, done
end

function step!(env::RandomWalk, action)
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

function available_actions(env::RandomWalk)
    return [1, 2]
end
