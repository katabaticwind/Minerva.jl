using Minerva, Flux
using Flux: mse

Q = Chain(Dense(4, 2))
loss(p, y) = mse(p, y)
opt = ADAM(0.01)
agent = DeepQAgent(Q, loss, opt, 10, 2, 0.01)
env = CartPole()
history = train!(agent, env, max_episodes = 10000)
