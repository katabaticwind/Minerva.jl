using Flux
using Flux: mse, back!
using Flux.Optimise: _update_params!
using Statistics
using Formatting


"""
HillClimber

    Agent attempts to improve by random parameter search.

    Run `N` epsiodes using a random parameterization of `Q` and record the average score. Next, make a random update to the parameterization, and run another `N` episodes to calculate the fitness of the new `Q`. If the new `Q` is fitter than the old `Q`, switch to the new `Q`. Otherwise, generate a new random update and repeat.
```
"""
mutable struct HillClimber
    W
    σ
end

HillClimber(A, S) = HillClimber(randn(A, S), 0.1)

function action(agent::HillClimber, env, ϵ = 0.0)
    if rand() < ϵ
        return rand(env.action_space)  # explore
    else
        return argmax(agent.W * env.state)  # exploit
    end
end

function perturb!(agent)
    W_ = deepcopy(agent.W)
    agent.W .+= agent.σ * randn(size(agent.W))
    return W_
end

function run_episode!(agent::HillClimber, env; ϵ = 0.05)
    total_reward = 0.0
    steps = 0
    s′, done = reset!(env)
    while !done
        # render(env)
        a = action(agent, env, ϵ)
        s = deepcopy(s′)
        s′, r, done, info = step!(env, a)
        # println("s = $s, a = $a, r = $r, s′ = $s′, done = $done")
        total_reward += r
        steps += 1
        update!(agent)
    end
    return total_reward, steps
end

function evaluate(agent::HillClimber, env; n = 100)
    history = []
    for i in 1:n
        total_reward = 0.0
        steps = 0
        s′, done = reset!(env)
        while !done
            a = action(agent, env)  # no exploration here (ϵ = 0.0)
            s = deepcopy(s′)
            s′, r, done, info = step!(env, a)
            total_reward += r
            steps += 1
            # println("s = $s, a = $a, r = $r, s′ = $s′, done = $done")
        end
        # @info "Finished episode $i" total_reward steps
        push!(history, (total_reward=total_reward, steps=steps))
    end
    return mean([e.total_reward for e in history])
end

function train!(agent::HillClimber, env; max_episodes = Inf)
    init_memory!(agent, env)
    @info "Training agent..."
    ϵ = ϵ0
    history = []
    iter = 0
    improving = true
    fitness = evaluate(agent, env)
    while improving && episodes < max_episodes
        iter += 1
        W = perturb!(agent)
        fitness = evaluate(agent, env)
        fitness < agent.fitness && revert!(agent, W)
        @info format("iter: {}, score: {}, epsilon: {}", episodes, fitness)
    end
    return history
end
