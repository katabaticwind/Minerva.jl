using Minerva

env = MountainCar()
s, done = reset!(env)
@info s, done
for step in 1:env.maxsteps
    a = rand(env.action_space)
    s′, r, done, info = step!(env, a)
    @info step, s′, r, done, info
    done && break
end
