using Minerva, Flux

"""
    Softmax()
"""
struct Softmax end

(a::Softmax)(x::AbstractArray) = softmax(x)

Base.show(io::IO, l::Softmax) = print(io, "Softmax()")


batch_size = 1
discount_rate = 1.00

model = Chain(
    Dense(4, 1024, relu),
    Dense(1024, 512, relu),
    Dense(512, 2),
    Softmax()
)
opt = Descent(1e-3)
agent = VanillaPG(model, opt, batch_size, discount_rate)
env = CartPole()
recorder = train!(agent, env, max_episodes = 10000)
evaluate(agent, env, 1, rendered = true)
