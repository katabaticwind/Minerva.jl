using Minerva

env = RandomEnvironment()
_ = reset!(env)
while true
    a = rand(env.action_space)
    s′, r, done, info = step!(env, a)
    @info s′, r, done, info
    done && break
end
