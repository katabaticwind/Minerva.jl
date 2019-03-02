using Minerva, Flux
using Flux: mse

# TODO: decay learning rate?
# TODO: add "do nothing" action?

max_memory = 1000
batch_size = 32
discount_rate = 1.00
update_episodes = 5  # epsiodes

Q = Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, 2))
loss(x, y) = mse(x, y)
opt = RMSProp(0.001)
agent = DeepQAgent(Q, loss, opt, max_memory, batch_size, discount_rate, update_episodes)
env = CartPole()
history = train!(agent, env, max_episodes = 1000)
evaluate(agent, env, n = 1, rendered = true)
