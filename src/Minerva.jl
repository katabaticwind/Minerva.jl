module Minerva

export RandomWalk,
       CartPole,
       MountainCar,
       reset!,
       step!

include("environment.jl")
include("environments/cartpole.jl")
include("environments/mountaincar.jl")

export DeepQAgent,
       action,
       train!,
       evaluate

include("agent.jl")
include("agents/deep_q.jl")


end # module
