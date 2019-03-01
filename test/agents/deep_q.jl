using Minerva, Flux
using Flux: mse

max_memory = 1000
batch_size = 32
discount_rate = 0.00
update_steps = 100

Q = Chain(Dense(4, 16, relu), Dense(16, 2))
loss(x, y) = mse(x, y)
opt = ADAM(0.0001)
agent = DeepQAgent(Q, loss, opt, max_memory, batch_size, discount_rate, update_steps)
env = CartPole()
history = train!(agent, env, max_episodes = 10000)
