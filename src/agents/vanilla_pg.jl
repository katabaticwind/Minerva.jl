using Flux
using Flux: mse, back!
using Flux.Optimise: _update_params!
using Statistics
using Formatting
using Random


# TODO: system for saving snapshots of the agent that can be re-loaded
# TODO: run_episode gets path to save gif to
# TODO: passing kwargs through as dicts?


"""
VanillaPG
"""
mutable struct VanillaPG
    model  # model
    opt  # optimizer ("watching" model)
    batch_size  # episodes per batch
    γ  # discount rate
end

function action(agent::VanillaPG, env::AbstractEnvironment, ϵ = 0.0)
    if rand() < ϵ
        return rand(env.action_space)  # explore
    else
        return argmax(agent.model(env.state))  # exploit
    end
end

function compute_loss(agent::VanillaPG, history, reward)
    S = hcat([o.s for o in history]...)
    a = [o.a for o in history]
    return mean(reward .* agent.model(S)[CartesianIndex.(a, axes(S, 2))])  # assume model parameterizes **log** or policy???
end

function update!(agent::VanillaPG, loss)
    loss = mean(loss)
    back!(loss)
    _update_params!(agent.opt, Flux.params(agent.model))
end

function run_episode(agent::VanillaPG, env::AbstractEnvironment; ϵ = 0.05, rendered = false, verbose = false)
    total_reward = 0.0
    steps = 0
    s′, done = reset!(env)
    rendered && render(env)
    history = []
    while !done
        a = action(agent, env, ϵ)
        s = deepcopy(s′)
        s′, r, done, info = step!(env, a)
        total_reward += r
        push!(history, (s=s, a=a))
        steps += 1
        verbose && @info "s = $s, a = $a, r = $r, s′ = $s′, done = $done"
        steps == env.maxsteps && break
    end
    verbose && @info "episode finished in $steps timesteps"
    loss = compute_loss(agent, history, total_reward)
    return total_reward, steps, loss
end

function evaluate(agent::VanillaPG, env::AbstractEnvironment, n = 100; ϵ = 0.05, verbose = false, rendered = false)
    env.rendered = rendered
    history = []
    @progress for i = 1:n
        total_reward, steps, _ = run_episode(agent, env, ϵ = ϵ, verbose = verbose)
        push!(history, (total_reward = total_reward, steps = steps))
    end
    total_reward = mean([e.total_reward for e in history])
    steps = mean([e.steps for e in history])
    env.rendered = false
    return total_reward, steps
end

function train!(agent::VanillaPG, env::AbstractEnvironment; max_episodes = Inf, eval_freq = 1, eval_episodes = 100, eval_ϵ = 0.05, ϵ_schedule = nothing, verbose = false, patience = 5)

    @info "training agent..."
    recorder = ExperimentRecorder("./log-$(randstring(5))")
    current_best_score = -Inf
    n_evals_since_best_score = 0
    improving = true
    batch_size = agent.batch_size
    batches = 0
    episodes = 0
    ϵ = ϵ_schedule(0)

    while improving && episodes < max_episodes

        loss = Array{Flux.Tracker.TrackedReal{Float64}, 1}(undef, batch_size)
        for i = 1:batch_size

            # training episode
            total_reward, steps, loss[i] = run_episode(agent, env, ϵ = ϵ)
            episodes += 1
            ϵ = ϵ_schedule(episodes)
            verbose && @info "<training status message>"
            message = (episode=episodes, score=total_reward, steps=steps, epsilon=ϵ)
            stamp!(recorder, message, :train)

        end
        batches += 1

        # network update
        update!(agent, loss)

        # fitness evaluation
        if batches % eval_freq == 0
            total_reward, steps = evaluate(agent, env, eval_episodes, ϵ = eval_ϵ)
            # @info "<testing status message>"
            @info "episode: $episodes, score: $total_reward"
            message = (episode=episodes, score=total_reward, steps=steps, epsilon=eval_ϵ)
            stamp!(recorder, message, :test)
            # run_episode(agent, env, rendered = true)  # to *see* how it's doing...

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
    # return history
end
