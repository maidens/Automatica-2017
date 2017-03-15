module DynamicProgramming

using Interpolations, JLD

export dp, dp_rollout, dp_reduced, dp_rollout_reduced

# index into a grid defined by a tuple of FloatRange
function index_into(grid::Tuple, grid_size::Tuple, index::Int64)
    return convert(Array{Float64, 1}, [grid[j][k] for (j, k) in enumerate(ind2sub(grid_size, index))])
end

# compute grid size as a tuple = (size x1, size x2, ..., size xn)
#             as well as an int = size x1 * size x2 * ... * size xn
function grid_size(grid::Tuple)
    tuple = ([length(l) for l in grid]...)
    int = 1
    for t in tuple
        int *= t
    end
    return tuple, int
end

# compute the step in reduced coordinates
#    requires function rho describing invariants and function rho_bar_inverse describing inverse invariants
#    see the paper "Symmetry reduction for dynamic programming" by John Maidens, Axel Barrau, Silvere Bonnabel, Murat Arcak

function first_step_reduced(N::Int64, x::Array{Float64, 1}, ugrid::Tuple, theta::Array{Float64, 1},
             f::Function, g::Function, g_terminal::Function, rho::Function, rho_bar_inverse::Function)
    J_star = -Inf
    ugrid_length_tuple, ugrid_length_total = grid_size(ugrid)
    for i=1:ugrid_length_total
        u = index_into(ugrid, ugrid_length_tuple, i)
        J_u = g(N, rho_bar_inverse(x), u, theta) + g_terminal(f(N, rho_bar_inverse(x), u, theta))
        if J_u > J_star
            J_star = J_u
        end
    end
    return J_star
end

function step_reduced(k::Int64, x::Array{Float64, 1}, ugrid::Tuple, theta::Array{Float64, 1},
            V, f::Function, g::Function, rho::Function, rho_bar_inverse::Function)
    J_star = -Inf
    ugrid_length_tuple, ugrid_length_total = grid_size(ugrid)
    for i=1:ugrid_length_total
        u = index_into(ugrid, ugrid_length_tuple, i)
        J_u = g(k, rho_bar_inverse(x), u, theta) + V[rho(f(k, rho_bar_inverse(x), u, theta))...]
        if J_u > J_star
            J_star = J_u
        end
    end
    return J_star
end

# compute optimal value function with dynamic programming algorithm in reduced coorinates using a parallel loop
#    requires function rho describing invariants and function rho_bar_inverse describing inverse invariants
#    see the paper "Symmetry reduction for dynamic programming" by John Maidens, Axel Barrau, Silvere Bonnabel, Murat Arcak
function dp_reduced(f::Function, phi::Function, g_terminal::Function, rho::Function, rho_bar_inverse::Function, ugrid::Tuple, xgrid::Tuple,
            theta0::Array{Float64, 1}, N::Int64)
    xgrid_length_tuple, xgrid_length_total = grid_size(xgrid)
    println("====== Constructing array J ======")
    println("J is size ", xgrid_length_tuple, " by ", N)
    J = SharedArray(Float64, xgrid_length_tuple..., N)
    println("====== Array J constructed  ======")

    # initialize J at final time
    println("Step k = ", N)
    @sync @parallel for i=1:xgrid_length_total
        x_i = index_into(xgrid, xgrid_length_tuple, i)
        J[ind2sub(xgrid_length_tuple, i)..., N] = first_step_reduced(N, x_i, ugrid, theta0, f, phi, g_terminal, rho, rho_bar_inverse)
    end

    # loop over all previous times
    for k = N-1:-1:1
        save("J_$k.jld", "J", J)
        println("Step k = ", k)
        V = interpolate(xgrid, reshape(slicedim(J, length(xgrid_length_tuple)+1, k+1), xgrid_length_tuple), Gridded(Linear()))
        V = extrapolate(V, -Inf) # ensures that V returns -Inf if x is outside bounds
        @sync @parallel for i=1:xgrid_length_total
            x_i = index_into(xgrid, xgrid_length_tuple, i)
            J[ind2sub(xgrid_length_tuple, i)..., k] = step_reduced(k, x_i, ugrid, theta0, V, f, phi, rho, rho_bar_inverse)
        end
    end
    return J
end


# TODO: currently assumes dynamic_xgrid has the same number of grid points at each time (though at different locations)
# function dp_reduced_dynamic_xgrid(f::Function, phi::Function, g_terminal::Function, rho::Function, rho_bar_inverse::Function, ugrid::Tuple, dynamic_xgrid::Tuple,
#            theta0::Array{Float64, 1}, N::Int64)
#    xgrid_length_tuple, xgrid_length_total = grid_size(dynamic_xgrid[1])
#    println("====== Constructing array J ======")
#    println("J is size ", xgrid_length_tuple, " by ", N)
#    J = SharedArray(Float64, xgrid_length_tuple..., N)
#    println("====== Array J constructed  ======")#

    # initialize J at final time
#    println("Step k = ", N)
#    @sync @parallel for i=1:xgrid_length_total
#        x_i = index_into(dynamic_xgrid[N], xgrid_length_tuple, i)
#        J[ind2sub(xgrid_length_tuple, i)..., N] = first_step_reduced(N, x_i, ugrid, theta0, f, phi, g_terminal, rho, rho_bar_inverse)
#    end

    # loop over all previous times
#    for k = N-1:-1:1
#        println("Step k = ", k)
#        V = interpolate(dynamic_xgrid[k+1], reshape(slicedim(J, length(xgrid_length_tuple)+1, k+1), xgrid_length_tuple), Gridded(Linear()))
#        V = extrapolate(V, -Inf) # ensures that V returns -Inf if x is outside bounds
#        @sync @parallel for i=1:xgrid_length_total
#            x_i = index_into(dynamic_xgrid[k], xgrid_length_tuple, i)
#            J[ind2sub(xgrid_length_tuple, i)..., k] = step_reduced(k, x_i, ugrid, theta0, V, f, phi, rho, rho_bar_inverse)
#        end
#    end
#    return J
#end


# shorthand for dynamic programming without symmetry reduction
function dp(f::Function, phi::Function, g_terminal::Function, ugrid::Tuple, xgrid::Tuple, theta0::Array{Float64, 1}, N::Int64)
    return dp_reduced(f, phi, g_terminal, identity, identity, ugrid, xgrid, theta0, N)
end

# compute optimal input at x
function step_u_reduced(k::Int64, x::Array{Float64, 1}, ugrid::Tuple, theta::Array{Float64, 1}, V, f::Function, g::Function, rho::Function)
    J_star = -Inf
    ugrid_length_tuple, ugrid_length_total = grid_size(ugrid)
    u_star = zeros(length(ugrid_length_tuple))
    for i=1:ugrid_length_total
        u = index_into(ugrid, ugrid_length_tuple, i)
        J_u = g(k, x, u, theta) + V[rho(f(k, x, u, theta))...]
        if J_u > J_star
            J_star = J_u
            u_star = u
        end
    end
    return u_star
end

# rollout a trajectory starting at x0 using the value function J
function dp_rollout_reduced_stochastic(J, x0::Array{Float64, 1}, f::Function, phi::Function, rho::Function,
                    ugrid::Tuple, xgrid::Tuple, theta0::Array{Float64, 1}, N::Int64, sigma::Float64)
    x = zeros((length(x0), N+1))
    u = zeros((length(ugrid), N))
    x[:, 1] = x0
    xgrid_length_tuple = ([length(l) for l in xgrid]...)
    for k=1:N
        V = interpolate(xgrid, reshape(slicedim(J, length(xgrid_length_tuple)+1, k), xgrid_length_tuple), Gridded(Linear()))
        V = extrapolate(V, -Inf) # ensures that V returns -Inf if x is outside bounds
        u[:, k] = step_u_reduced(k, x[:, k], ugrid, theta0, V, f, phi, rho)
        n = [0; 0; sigma*randn(1); 0; 0; sigma*randn(1)] # define noise 
        x[:, k+1] = f(k, x[:, k], u[:, k], theta0) + n
    end
    return x, u
end

function dp_rollout_reduced(J, x0::Array{Float64, 1}, f::Function, phi::Function, rho::Function,
                    ugrid::Tuple, xgrid::Tuple, theta0::Array{Float64, 1}, N::Int64)
    return dp_rollout_reduced_stochastic(J, x0, f, phi, rho, ugrid, xgrid, theta0, N, 0.)
end

# shorthand for rollout without symmetry reduction
function dp_rollout(J, x0::Array{Float64, 1}, f::Function, phi::Function,
                    ugrid::Tuple, xgrid::Tuple, theta0::Array{Float64, 1}, N::Int64)
    return dp_rollout_reduced(J, x0, f, phi, identity, ugrid, xgrid, theta0, N)
end

end # Module
