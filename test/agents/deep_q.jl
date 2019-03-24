using Minerva, Flux
using Flux: mse

# TODO: BatchNorm?
# TODO: Dropout?
# TODO: Regularization?

# agent settings
max_memory = 1000
discount_rate = 1.00
update_episodes = 5

# network settings
batch_size = 32
learning_rate = 1e-2

Q = Chain(
    Dense(4, 64),
    Dense(64, 64),
    Dense(64, 2)
)
agent = DeepQAgent(Q, learning_rate)
env = CartPole()
ϵ_schedule = StepDecay(1.0, 0.05, 0.1, 1)
recorder = train!(agent, env, max_episodes = 500, ϵ_schedule = ϵ_schedule)
evaluate(agent, env, 1, ϵ = 0.05, rendered = true)
