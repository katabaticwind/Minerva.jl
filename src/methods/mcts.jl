mutable struct Tree
    state
    action
    value
    count
    depth
    terminal
    children
    parent
end

EPSILON = 0.01

function transition(state, action)

    next_state = nothing
    reward = nothing
    done = nothing

    # next_state
    if state < 0.5 - EPSILON && action == 1
        next_state = 0.6 * rand()  # [0, 0.6]
        reward = 1.
    elseif state < 0.5 - EPSILON && action == 2
        next_state = 0.75 * rand()  # [0, 0.75]
        reward = 0.
    elseif state > 0.5 + EPSILON && action == 2
        next_state = 0.4 + 0.5 * rand()  # [0.4, 1.0]
        reward = 1.
    elseif state > 0.5 + EPSILON && action == 1
        next_state = 0.25 + 0.75 * rand()  # [0.25, 1.0]
        reward = 0.
    end

    # done
    if next_state > 0.5 - EPSILON && next_state < 0.5 + EPSILON
        done = true
    else
        done = false
    end

    # return
    return next_state, reward, done
end

ACTIONS = [1, 2]

default_policy(state) = rand(ACTIONS)

function score(node, alpha = 1.0)
    avg_value = node.value / node.count
    rel_freq = 2 * alpha * sqrt(2 * log(node.parent.count) / node.count)
    return avg_value + rel_freq
end

function uct_search(init_state, max_depth = 20, max_steps = 100)
    root = Tree(init_state, nothing, 0.0, 0, 0, false, Array{Tree,1}(), nothing)
    steps = 0
    node = root
    while node.depth < max_depth && steps < max_steps
        value = evaluate(node, default_policy)
        backup(node, value)
        node = tree_policy(root)
        steps += 1
        println("step = $steps, depth = $(node.depth)")
    end
    return argmax([n.value for n in root.children])
end

function tree_policy(node)
    """Find the next leaf to evaluate.

        If the current node is expandable, return a node for an untried action. Else, choose the "best" child node and repeat until you arrive at a terminal node.
    """
    while ~node.terminal
        child = expand(node)
        if child == nothing
            node = best_child(node)
        else
            return child
        end
    end
    return node
end

function expand(node)
    available_actions = setdiff(ACTIONS, Set([n.action for n in node.children]))
    length(available_actions) == 0 && return nothing  # node *not* expandable
    action = rand(available_actions)
    state, _, done = transition(node.state, action)
    child = Tree(state, action, 0.0, 0, node.depth + 1, done, Array{Tree,1}(), node)
    push!(node.children, child)
    return child
end

best_child(node) = node.children[argmax(score.(node.children))]

function evaluate(node, policy)
    """Evaluate a node by running `policy` until end of episode."""
    value = 0.0
    state = node.state
    done = node.terminal
    while ~done
        action = policy(state)
        state, reward, done = transition(state, action)
        value += reward
    end
    return value
end

function backup(node, value)
    """Update value of each node by adding `value` and incrementing count."""
    while node != nothing
        node.count += 1
        node.value += value
        node = node.parent
    end
end

init_state = rand()
opt_action = uct_search(init_state)
