module Minerva

using Plots, Juno

include("environment.jl")
include("environments/cartpole.jl")
include("environments/mountaincar.jl")

export RandomWalk,
       CartPole,
       MountainCar,
       reset!,
       step!,
       render

include("agent.jl")
include("agents/deep_q.jl")
include("agents/vanilla_pg.jl")

export DeepQAgent,
       VanillaPG,
       action,
       train!,
       evaluate

include("schedulers.jl")

export StepDecay,
       ExpDecay

include("recorders.jl")

export ExperimentRecorder,
       stamp!

end # module
