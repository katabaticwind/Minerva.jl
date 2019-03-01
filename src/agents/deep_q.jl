using Flux
using Flux: mse, back!
using Flux.Optimise: _update_params!
using Statistics
using Formatting


"""
DeepQAgent

- `Q`: the action-value function, i.e., Q(s) -> Q(s, a) for all actions a ∈ A.
- `memory`: buffer storing transitions `(s, a, r, s′)`.
- `γ`: reward discount rate.

# Example
```julia-repl
Q = Chain(
Conv((3, 3), 1=>32, relu),
Conv((3, 3), 32=>32, relu),
x -> maxpool(x, (2,2)),
Conv((3, 3), 32=>16, relu),
x -> maxpool(x, (2,2)),
Conv((3, 3), 16=>10, relu),
x -> reshape(x, :, size(x, 4)),
Dense(90, 10),
softmax)
loss(x, y) = mse(x, y)
opt = ADAM()
agent = DeepQAgent(Q, loss, opt, 1000, 10, 0.01)
```
"""
mutable struct DeepQAgent
    Q  # action-value
    Q_target
    loss
    opt
    memory
    max_memory
    batch_size
    γ  # discount rate
    update_steps
end

DeepQAgent(Q, loss, opt, mm) = DeepQAgent(Q, deepcopy(Q), loss, opt, [], mm, 10, 0.0, 1000)
DeepQAgent(Q, loss, opt, mm, bs) = DeepQAgent(Q, deepcopy(Q), loss, opt, [], mm, bs, 0.0, 1000)
DeepQAgent(Q, loss, opt, mm, bs, γ, us) = DeepQAgent(Q, deepcopy(Q), loss, opt, [], mm, bs, γ, us)

function action(agent::DeepQAgent, env::AbstractEnvironment, ϵ = 0.0)
    if rand() < ϵ
        return rand(env.action_space)  # explore
    else
        return argmax(agent.Q(env.state))  # exploit
    end
end

function update!(agent::DeepQAgent)
    mem_batch = rand(agent.memory, agent.batch_size)
    p = map(sars -> estimate(sars, agent), mem_batch)
    y = map(sars -> target(sars, agent), mem_batch)
    l = agent.loss(p, y)
    back!(l)
    _update_params!(agent.opt, Flux.params(agent.Q))
end

function update_target(agent::DeepQAgent)
    agent.Q_target = deepcopy(agent.Q)
end

function estimate(sars, agent)
    s, a, _, _, done = sars
    q = agent.Q(s)
    return X = q[a, :][1]
end

function target(sars, agent)
    _, _, r, s′, done = sars
    if done
        return r
    else
        return r + agent.γ * maximum(agent.Q_target(s′).data)
    end
end

function init_memory!(agent::DeepQAgent, env::AbstractEnvironment)
    @info "Initializing agent memory..."
    if length(agent.memory) > 0
        agent.memory = []
        @info "> cleared existing memory"
    end
    while length(agent.memory) < agent.max_memory
        s′, done = reset!(env)
        while !done
            a = action(agent, env)
            s = deepcopy(s′)
            s′, r, done, info = step!(env, a)
            # println("s = $s, a = $a, r = $r, s′ = $s′, done = $done")
            length(agent.memory) == agent.max_memory && return
            pushfirst!(agent.memory, [s, a, r, s′, done])
        end
    end
end

function run_episode!(agent::DeepQAgent, env::AbstractEnvironment; ϵ = 0.05)
    total_reward = 0.0
    steps = 0
    s′, done = reset!(env)
    while !done
        # render(env)
        a = action(agent, env, ϵ)
        s = deepcopy(s′)
        s′, r, done, info = step!(env, a)
        # println("s = $s, a = $a, r = $r, s′ = $s′, done = $done")
        length(agent.memory) == agent.max_memory && pop!(agent.memory)
        pushfirst!(agent.memory, [s, a, r, s′, done])
        total_reward += r
        steps += 1
        update!(agent)
        if steps % agent.update_steps == 0
            update_target(agent)
        end
    end
    # @info "Episode finished after $steps timesteps"
    return total_reward, steps
end

function evaluate(agent::DeepQAgent, env::AbstractEnvironment; n = 100)
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

function train!(agent::DeepQAgent, env::AbstractEnvironment; max_episodes = Inf, ϵ0 = 1.00)
    init_memory!(agent, env)
    @info "Training agent..."
    ϵ = ϵ0
    history = []
    episodes = 0
    improving = true
    while improving && episodes < max_episodes
        total_reward, steps = run_episode!(agent, env, ϵ = ϵ)
        ϵ = stepdecay(episodes, ϵ0, 0.01, 1000, 0.1)
        push!(history, (total_reward=total_reward, steps=steps))
        episodes += 1
        if episodes % 100 == 0
            score = evaluate(agent, env, n = 100)
            @info format("episodes: {}, score: {}, epsilon: {}", episodes, score, ϵ)
        end
    end
    return history
end

stepdecay(t, max_x, min_x, nsteps, stepsize) = max(max_x - floor(t / nsteps) * stepsize, min_x)
