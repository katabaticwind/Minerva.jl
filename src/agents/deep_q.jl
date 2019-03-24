using Flux
using Flux: mse, back!
using Flux.Optimise: _update_params!
using Statistics
using Formatting
using Random


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
    update_episodes
end

function DeepQAgent(Q, lr)
    Q_target = deepcopy(Q)
    loss = mse
    opt = RMSProp(lr)
    memory = []
    max_memory = 1000
    batch_size = 32
    γ = 1.0
    update_episodes = 5
    return DeepQAgent(Q,
                      Q_target,
                      loss,
                      opt,
                      memory,
                      max_memory,
                      batch_size,
                      γ,
                      update_episodes)
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

function update_target!(agent::DeepQAgent)
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

# function update!(agent::DeepQAgent)
#     mem_batch = rand(agent.memory, agent.batch_size)
#     S = hcat([m[1] for m in mem_batch]...)
#     A = hcat([m[2] for m in mem_batch]...)
#     R = hcat([m[3] for m in mem_batch]...)
#     S_ = hcat([m[4] for m in mem_batch]...)
#     done = hcat([m[5] for m in mem_batch]...)
#     target = R .+ agent.γ .* .!done .* maximum(agent.Q_target(S_).data, dims = 1)
#     estimate = agent.Q(S)[CartesianIndex.(A', axes(S, 2))]
#     loss = agent.loss(estimate, target)
#     back!(loss)
#     _update_params!(agent.opt, Flux.params(agent.Q))
# end

function init_memory!(agent::DeepQAgent, env::AbstractEnvironment)
    @info "Initializing agent memory..."
    if length(agent.memory) > 0
        agent.memory = []
        @info "> cleared existing memory"
    end
    while length(agent.memory) < agent.max_memory
        s′, done = reset!(env)
        while !done
            a = action(agent, env, 1.0)  # random action
            s = deepcopy(s′)
            s′, r, done, info = step!(env, a)
            length(agent.memory) == agent.max_memory && return
            pushfirst!(agent.memory, [s, a, r, s′, done])
        end
    end
end

"""
run_episode!(agent, env; <kwargs>)

    Train `agent` for one episode in `env` following ϵ-greedy policy.

    # Arguments
    - `ϵ`: probability of performing a non-greedy action.
"""
function run_episode!(agent::DeepQAgent, env::AbstractEnvironment; ϵ = 0.05, rendered = false, verbose = false)
    total_reward = 0.0
    steps = 0
    s′, done = reset!(env)
    env.rendered = rendered
    while !done
        a = action(agent, env, ϵ)
        s = deepcopy(s′)
        s′, r, done, info = step!(env, a)
        verbose && @info "s = $s, a = $a, r = $r, s′ = $s′, done = $done"
        length(agent.memory) == agent.max_memory && pop!(agent.memory)
        pushfirst!(agent.memory, [s, a, r, s′, done])
        total_reward += r
        steps += 1
        update!(agent)
    end
    env.rendered = false
    verbose && @info "episode finished in $steps timesteps"
    return total_reward, steps
end

"""
run_episode(agent, env; <kwargs>)

    Evaluate `agent` for one epsiode in `env` following ϵ-greedy policy.

    **NOTE** This method does **not** update the agent.

    # Arguments
    - `ϵ`: probability of performing a non-greedy action.
"""
function run_episode(agent::DeepQAgent, env::AbstractEnvironment; ϵ = 0.05, rendered = false, verbose = false)
    total_reward = 0.0
    steps = 0
    s′, done = reset!(env)
    env.rendered = rendered
    while !done
        a = action(agent, env, ϵ)
        s = deepcopy(s′)
        s′, r, done, info = step!(env, a)
        total_reward += r
        steps += 1
        verbose && @info "s = $s, a = $a, r = $r, s′ = $s′, done = $done"
        if steps == env.maxsteps
            @info "> agent reached end of episode!"
            break
        end
    end
    env.rendered = false
    verbose && @info "episode finished in $steps timesteps"
    return total_reward, steps
end

function evaluate(agent::DeepQAgent, env::AbstractEnvironment, n = 100; ϵ = 0.05, rendered = false, verbose = false)
    history = []
    @progress for i = 1:n
        total_reward, steps = run_episode(agent, env, ϵ = ϵ, rendered = rendered, verbose = verbose)
        push!(history, (total_reward = total_reward, steps = steps))
    end
    total_reward = mean([e.total_reward for e in history])
    steps = mean([e.steps for e in history])
    return total_reward, steps
end

"""
train!(agent, env; <kwargs>)

    Train `agent` on `env`.

    # Arguments
    - `max_episodes`: maximum number of episodes to train the agent.
    - `eval_freq`: number of episodes between fitness evaluations.
    - `ϵ_schedule`: determines ϵ-annealing based on number of epsiodes.

    # Example
    ```julia-repl
    ϵ_schedule = StepDecay(1.0, 0.05, 0.1, 50)
    agent = DeepQAgent(...)
    env = CartPole(...)
    train!(agent, env, eval_freq = 20, ϵ_schedule = ϵ_schedule)
    ```
"""
function train!(agent::DeepQAgent, env::AbstractEnvironment; max_episodes = Inf, eval_freq = 20, eval_episodes = 100, eval_ϵ = 0.05, ϵ_schedule = nothing, verbose = false, patience = 10)

    init_memory!(agent, env)
    @info "training agent..."
    episodes = 0
    recorder = ExperimentRecorder("./log-$(randstring(5))")
    current_best_score = -Inf
    n_evals_since_best_score = 0
    improving = true
    ϵ = ϵ_schedule(0)
    while improving && episodes < max_episodes
        # training episode
        total_reward, steps = run_episode!(agent, env, ϵ = ϵ)
        episodes += 1
        ϵ = ϵ_schedule(episodes)
        verbose && @info "<training status message>"
        message = (episode=episodes, score=total_reward, steps=steps, epsilon=ϵ)
        stamp!(recorder, message, :train)

        agent.opt.eta = 0.99 * agent.opt.eta

        # *target* network update
        verbose && @info "updating target network..."
        if episodes % agent.update_episodes == 0
            update_target!(agent)
        end

        # fitness evaluation
        if episodes % eval_freq == 0
            score, steps = evaluate(agent, env, eval_episodes, ϵ = eval_ϵ)
            @info "episode: $episodes, score: $score, lr: $(agent.opt.eta), ϵ: $ϵ"
            message = (episode=episodes, score=score, steps=steps, epsilon=eval_ϵ)
            stamp!(recorder, message, :test)
            # evaluate(agent, env, 1, ϵ = eval_ϵ, rendered = true)

            # snapshots
            if score > current_best_score
                @info "> new best score achieved!"
                current_best_score = score
                n_evals_since_best_score = 0
                # save a snapshot of agent...
            else
                n_evals_since_best_score += 1
                n_evals_since_best_score == patience && break
            end
        end
    end
    verbose && @info "agent finished training in $episodes episodes"
    return recorder
end
