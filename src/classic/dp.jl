mutable struct GridWorld
    n
    rewards
    probabilities
end

function GridWorld(n)
    p = Dict(:rewards => ones(n * n, 4, 1), :states => zeros(n * n, 4, n * n))
    r = [-1]
    for i in 1:n
        for j in 1:n
            s_index = cartesian_to_linear(i, j, n)
            if index_to_state(s_index, n) == "x"
                # pass
            elseif index_to_state(s_index, n) == "top-right"
                p[:states][s_index, 1, cartesian_to_linear(i, j - 1, n)] = 1.0
                p[:states][s_index, 2, cartesian_to_linear(i, j, n)] = 1.0  # fixed
                p[:states][s_index, 3, cartesian_to_linear(i, j, n)] = 1.0  # fixed
                p[:states][s_index, 4, cartesian_to_linear(i + 1, j, n)] = 1.0
            elseif index_to_state(s_index, n) == "bottom-left"
                p[:states][s_index, 1, cartesian_to_linear(i, j, n)] = 1.0  # fixed
                p[:states][s_index, 2, cartesian_to_linear(i, j + 1, n)] = 1.0
                p[:states][s_index, 3, cartesian_to_linear(i - 1, j, n)] = 1.0
                p[:states][s_index, 4, cartesian_to_linear(i, j, n)] = 1.0  # fixed
            elseif index_to_state(s_index, n) == "left"
                p[:states][s_index, 1, cartesian_to_linear(i, j, n)] = 1.0  # fixed
                p[:states][s_index, 2, cartesian_to_linear(i, j + 1, n)] = 1.0
                p[:states][s_index, 3, cartesian_to_linear(i - 1, j, n)] = 1.0
                p[:states][s_index, 4, cartesian_to_linear(i + 1, j, n)] = 1.0
            elseif index_to_state(s_index, n) == "right"
                p[:states][s_index, 1, cartesian_to_linear(i, j - 1, n)] = 1.0
                p[:states][s_index, 2, cartesian_to_linear(i, j, n)] = 1.0  # fixed
                p[:states][s_index, 3, cartesian_to_linear(i - 1, j, n)] = 1.0
                p[:states][s_index, 4, cartesian_to_linear(i + 1, j, n)] = 1.0
            elseif index_to_state(s_index, n) == "top"
                p[:states][s_index, 1, cartesian_to_linear(i, j - 1, n)] = 1.0
                p[:states][s_index, 2, cartesian_to_linear(i, j + 1, n)] = 1.0
                p[:states][s_index, 3, cartesian_to_linear(i, j, n)] = 1.0  # fixed
                p[:states][s_index, 4, cartesian_to_linear(i + 1, j, n)] = 1.0
            elseif index_to_state(s_index, n) == "bottom"
                p[:states][s_index, 1, cartesian_to_linear(i, j - 1, n)] = 1.0
                p[:states][s_index, 2, cartesian_to_linear(i, j + 1, n)] = 1.0
                p[:states][s_index, 3, cartesian_to_linear(i - 1, j, n)] = 1.0
                p[:states][s_index, 4, cartesian_to_linear(i, j, n)] = 1.0  # fixed
            else  index_to_state(s_index, n) == "o"
                p[:states][s_index, 1, cartesian_to_linear(i, j - 1, n)] = 1.00
                p[:states][s_index, 2, cartesian_to_linear(i, j + 1, n)] = 1.00
                p[:states][s_index, 3, cartesian_to_linear(i - 1, j, n)] = 1.00
                p[:states][s_index, 4, cartesian_to_linear(i + 1, j, n)] = 1.00
            end
        end
    end
    grid = Array{String}(undef, n, n)
    for i in eachindex(grid)
        grid[i] = index_to_state(i, n)
    end
    return GridWorld(n, r, p), grid
end

function index_to_state(index, n)
    """Return `state` in n x n GridWorld given index."""

    if (index == 1) || (index == n * n)
        return "x"
    elseif div(index - 1, n) == n - 1 && rem(index, n) == 1
        return "top-right"
    elseif div(index - 1, n) == 0 && rem(index, n) == 0
        return "bottom-left"
    elseif div(index - 1, n) == 0
        return "left"
    elseif div(index - 1, n) == n - 1
        return "right"
    elseif rem(index, n) == 1
        return "top"
    elseif rem(index, n) == 0
        return "bottom"
    else
        return "o"
    end
end

function cartesian_to_linear(i, j, n)
    return (j - 1) * n + i
end

function random_policy(n)
    """Create random policy matrix for GridWorld(n)"""
    return 0.25 * ones(n * n, 4)
end

function evaluate(pA, pR, pS, R, gamma, tol=1e-6)
    """Evaluate policy via dynamic programming.

        - `pA`: array of action probabilities, p(a | s).
        - `pR`: array of reward probabilities, p(r | s, a).
        - `pS`: array of action probabilities, p(s' | s, a).
        - `R`: array of reward values *corresponding* to p(r | s, a).

        # Returns
        - `V`: vector of policy values, V(s).
    """

    V = zeros(size(pA)[1])
    steps = 0
    delta = 0.0
    while true
        delta = 0.0
        for (v_idx, v) in enumerate(V)
            v_update = 0.0
            for (a_idx, pa) in enumerate(pA[v_idx, :])
                for (r_idx, pr) in enumerate(pR[v_idx, a_idx, :])
                    r = R[r_idx]
                    for (s_idx, ps) in enumerate(pS[v_idx, a_idx, :])
                        v_update += pa * pr * ps * (r + gamma * V[s_idx])
                    end
                end
            end
            V[v_idx] = v_update
            # println("V[$v_idx] = $(V[v_idx])")
            delta = max(delta, abs(v - v_update))  # update maximum change
        end
        steps += 1
        println("delta=$delta")
        if delta < tol
            break
        end
    end
    println("converged in $steps steps (delta=$delta)")
    return V
end

world, grid = GridWorld(4);
pA = random_policy(4);
pR = world.probabilities[:rewards];
pS = world.probabilities[:states];
R = world.rewards;
gamma = 1.0;
V = evaluate(pA, pR, pS, R, gamma);
