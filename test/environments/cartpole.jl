using Minerva

function run_episodes()
    env = CartPole()
    # render(env)
    scores = []
    for i = 1:1000
        _ = reset!(env)
        total_reward = 0
        while true
            a = rand(env.action_space)
            s′, r, done, info = step!(env, a)
            total_reward += r
            # @info step, s′, r, done, info
            done && break
        end
        push!(scores, total_reward)
    end
    return scores
end

# scores = run_episodes()

function sample_run()
    env = CartPole()
    _ = reset!(env)
    render(env)
    total_reward = 0
    steps = 0
    while true
        a = rand(env.action_space)
        s′, r, done, info = step!(env, a)
        total_reward += r
        steps += 1
        # @info step, s′, r, done, info
        done && break
    end
    @info steps
    return total_reward, steps
end

_ = sample_run()
