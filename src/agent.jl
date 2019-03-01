abstract type AbstractAgent end

function action(agent, env, ϵ) end
function update!(agent) end
function run_episode!(agent, env) end
function evaluate(agent, env, n = 100) end
function train!(agent, env, max_episodes = Inf) end
