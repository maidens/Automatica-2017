module LinearQuadraticSymmetryTests

using Base.Test

include("../DynamicProgramming.jl")

# horizon
N = 30

# initial condition (used to test policy rollout)
x0 = [1.; 0.]

# model parameters
theta0 = Array(Float64, 1)

# dynamics
A = eye(2)
B = eye(2)
function f(k::Int64, x::Array{Float64, 1}, u::Array{Float64, 1}, theta::Array{Float64, 1})
  return A*x + B*u
end

# reward function
Q = -[1 -1; -1 1]
R = zeros(2, 2)
function g(k::Int64, x::Array{Float64, 1}, u::Array{Float64, 1}, theta::Array{Float64, 1})
    return dot(x, Q*x) # + dot(u, R*u)
end

# terminal reward
function g_terminal(x::Array{Float64, 1})
    return dot(x, Q*x)
end

# reduction functions
function rho(x::Array{Float64, 1})
  return [x[2] - x[1]]
end

function rho_bar_inverse(x_bar::Array{Float64, 1})
  return [0; x_bar[1]]
end

# input grid
ugrid = (linspace(-0.1, 0.1, 3), linspace(-0.1, 0.1, 3))

# reduced state grid
xgrid_reduced = (linspace(-1, 1, 51),)

# test that a corresponding trajectory rollout agrees with analytic solution
epsilon = 0.05
J = DynamicProgramming.dp_reduced(f, g, g_terminal, rho, rho_bar_inverse, ugrid, xgrid_reduced, theta0, N)
x_rollout, u_rollout = DynamicProgramming.dp_rollout_reduced(J, x0, f, g, rho, ugrid, xgrid_reduced, theta0, N)
@test_approx_eq_eps([1.0 0.9 0.8 0.7 0.6 0.5; 0.0 0.1 0.2 0.3 0.4 0.5], x_rollout[:, 1:6], epsilon)
@test_approx_eq_eps([-0.1 -0.1 -0.1 -0.1 -0.1; 0.1 0.1 0.1 0.1 0.1], u_rollout[:, 1:5], epsilon)

end
