using Flux
using Flux: mse, back!
using Flux.Optimise: _update_params!
using Statistics

abstract type AbstractAgent end

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
    loss
    opt
    memory
    max_memory
    batch_size
    γ  # discount rate
end

DeepQAgent(Q, loss, opt, max_memory) = DeepQAgent(Q, loss, opt, [], max_memory, 10, 0.0)
DeepQAgent(Q, loss, opt, max_memory, batch_size) = DeepQAgent(Q, loss, opt, [], batch_size, 0.0)
DeepQAgent(Q, loss, opt, max_memory, batch_size, γ) = DeepQAgent(Q, loss, opt, [], max_memory, batch_size, γ)

function action(agent::DeepQAgent, env::AbstractEnvironment, ϵ = 0.0)
    if rand() < ϵ
        return rand(env.action_space)  # explore
    else
        return argmax(agent.Q(env.state))  # exploit
    end
end

function update!(agent::DeepQAgent)
    sars = rand(agent.memory, agent.batch_size)
    yhat = map(s -> estimate(s, agent), sars)
    y = map(s -> target(s, agent), sars)
    l = agent.loss(yhat, y)
    back!(l)
    _update_params!(agent.opt, Flux.params(agent.Q))
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
        return r + agent.γ * maximum(agent.Q(s′))
    end
end

function init_memory!(agent::DeepQAgent, env::AbstractEnvironment)
    println("Initializing agent memory...")
    if length(agent.memory) > 0
        agent.memory = []
        println("(cleared existing memory)")
    end
    while length(agent.memory) < agent.max_memory
        s′, done = reset!(env)
        while !done
          a = action(agent, env)
          s = deepcopy(s′)
          s′, r, done, info = step!(env, a)
          println("s = $s, a = $a, r = $r, s′ = $s′, done = $done")
          if length(agent.memory) == agent.max_memory
              println("done.")
              return
          end
          pushfirst!(agent.memory, [s, a, r, s′, done])
        end
    end
    println("done.")
end

function run_episode!(agent::DeepQAgent, env::AbstractEnvironment)
    total_reward = 0.0
    steps = 0
    s′, done = reset!(env)
    while !done
      # render(env)
      a = action(agent, env)
      s = deepcopy(s′)
      s′, r, done, info = step!(env, a)
      println("s = $s, a = $a, r = $r, s′ = $s′, done = $done")
      length(agent.memory) == agent.max_memory && pop!(agent.memory)
      pushfirst!(agent.memory, [s, a, r, s′, done])
      total_reward += r
      steps += 1
      update!(agent)
    end
    println("> Episode finished after $steps timesteps")
    return total_reward, steps
end

function evaluate(agent::DeepQAgent, env::AbstractEnvironment; n = 100)
    history = []
    for _ in 1:n
        total_reward = 0.0
        steps = 0
        s′, done = reset!(env)
        while !done
            a = action(agent, env)
            s = deepcopy(s′)
            s′, r, done, info = step!(env, a)
            total_reward += r
            steps += 1
            # println("s = $s, a = $a, r = $r, s′ = $s′, done = $done")
        end
        push!(history, (total_reward=total_reward, steps=steps))
    end
    return mean([e.total_reward for e in history])
end

function train!(agent::DeepQAgent, env::AbstractEnvironment; max_episodes = Inf)
    init_memory!(agent, env)
    println("Training agent...")
    history = []
    episodes = 0
    improving = true
    while improving && episodes < max_episodes
        total_reward, steps = run_episode!(agent, env)
        push!(history, (total_reward=total_reward, steps=steps))
        episodes += 1
        score = evaluate(agent, env, n = 1)
        println("> Score = $score")
    end
    println("done.")
    return history
end
