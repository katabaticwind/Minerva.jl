# using Flux
# using Flux: mse, back!
# using Flux.Optimise: _update_params!

include("./deep_q.jl")
include("./reboot.jl")

Q = Chain(Dense(1, 2))
loss(x, y) = mse(x, y)
opt = ADAM()
agent = DeepQAgent(Q, loss, opt, 10, 2, 0.01)
env = BasicEnvironment()
history = train!(agent, env, max_episodes = 2)
