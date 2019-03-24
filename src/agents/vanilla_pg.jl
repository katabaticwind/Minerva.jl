using Flux
using Flux: mse, back!
using Flux.Optimise: _update_params!
using Statistics
using Formatting
using Random
using StatsBase


"""
VanillaPG

    # Notes
    - VPG chooses a random action according to its policy function, π.
        - Can learn a (nearly) deterministic policy if that is optimal.
"""
mutable struct VanillaPG
    model  # model
    opt  # optimizer ("watching" model)
    batch_size  # episodes per batch
    γ  # discount rate
end

action(agent::VanillaPG, env::AbstractEnvironment) = sample(env.action_space, weights(agent.model(env.state)))

function compute_loss(agent::VanillaPG, states, actions, rewards)
    states = hcat(states...)
    index = CartesianIndex.(actions, axes(states, 2))
    return sum(cumsum(rewards) .* log.(agent.model(states)[index]))
end

function update!(agent::VanillaPG, loss)
    loss = -mean(loss)
    back!(loss)
    _update_params!(agent.opt, Flux.params(agent.model))
end

function run_episode(agent::VanillaPG, env::AbstractEnvironment; rendered = false, verbose = false)
    total_reward = 0.0
    steps = 0
    s′, done = reset!(env)
    rendered && render(env)
    states = []
    actions = []
    rewards = []
    while !done
        s = deepcopy(s′)
        a = action(agent, env)
        push!(states, s)
        push!(actions, a)
        s′, r, done, info = step!(env, a)
        steps += 1
        total_reward += r
        push!(rewards, agent.γ^steps * r)  # NOTE: **discounted** rewards
        verbose && @info "s = $s, a = $a, r = $r, s′ = $(s′), done = $done"
        steps == env.maxsteps && break
    end
    verbose && @info "episode finished in $steps timesteps"
    loss = compute_loss(agent, states, actions, rewards)
    return total_reward, steps, loss
end

function evaluate(agent::VanillaPG, env::AbstractEnvironment, n = 100; verbose = false, rendered = false)
    env.rendered = rendered
    history = []
    @progress for i = 1:n
        total_reward, steps, _ = run_episode(agent, env, verbose = verbose)
        push!(history, (total_reward = total_reward, steps = steps))
    end
    total_reward = mean([e.total_reward for e in history])
    steps = mean([e.steps for e in history])
    env.rendered = false
    return total_reward, steps
end

function train!(agent::VanillaPG, env::AbstractEnvironment; max_episodes = Inf, eval_freq = 1, eval_episodes = 100, verbose = false, patience = 5)

    @info "training agent..."
    recorder = ExperimentRecorder("./log-$(randstring(5))")
    current_best_score = -Inf
    n_evals_since_best_score = 0
    improving = true
    batch_size = agent.batch_size
    batches = 0
    episodes = 0

    while improving && episodes < max_episodes

        loss = Array{Flux.Tracker.TrackedReal{Float64}, 1}(undef, batch_size)
        for i = 1:batch_size
            # training episode
            total_reward, steps, loss[i] = run_episode(agent, env)
            episodes += 1
            verbose && @info "<training status message>"
            message = (episode=episodes, score=total_reward, steps=steps, epsilon=NaN)
            stamp!(recorder, message, :train)
        end

        # network update
        update!(agent, loss)
        batches += 1

        # agent.opt.eta = agent.opt.eta * 0.99

        # fitness evaluation
        if batches % eval_freq == 0
            total_reward, steps = evaluate(agent, env, eval_episodes)
            @info "episode: $episodes, score: $total_reward, η: $(agent.opt.eta)"
            message = (episode=episodes, score=total_reward, steps=steps, epsilon=NaN)
            stamp!(recorder, message, :test)

            # snapshots
            # if score > current_best_score
            #     current_best_score = score
            #     # save a snapshot of agent...
            # else
            #     n_evals_since_best_score += 1
            #     n_evals_since_best_score == patience && break
            # end
        end
    end
    verbose && @info "agent finished training in $episodes episodes"
    return recorder
end
