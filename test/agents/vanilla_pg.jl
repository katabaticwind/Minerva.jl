using Minerva, Flux

batch_size = 32
discount_rate = 1.00

model = Chain(Dense(4, 64, relu), Dense(64, 64, relu), Dense(64, 2))
opt = RMSProp(1e-3)
agent = VanillaPG(model, opt, batch_size, discount_rate)
env = CartPole()
ϵ_schedule = StepDecay(1.0, 0.05, 0.1, 50)
train!(agent, env, max_episodes = 1000, ϵ_schedule = ϵ_schedule)
evaluate(agent, env, 1, rendered = true)
