using Minerva, Flux

"""
    Softmax()
"""
struct Softmax end

(a::Softmax)(x::AbstractArray) = softmax(x)

Base.show(io::IO, l::Softmax) = print(io, "Softmax()")


batch_size = 32
discount_rate = 1.00

model = Chain(
    Dense(4, 64, leakyrelu),
    Dense(64, 2),
    Softmax()
)
opt = RMSProp(1e-3)
agent = VanillaPG(model, opt, batch_size, discount_rate)
env = CartPole()
recorder = train!(agent, env, max_episodes = 1000)
evaluate(agent, env, 1, rendered = true)
