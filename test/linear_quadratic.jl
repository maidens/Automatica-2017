module LinearQuadraticTests

using Base.Test

include("../DynamicProgramming.jl")

# horizon
N = 30

# initial condition (used to test policy rollout)
x0 = [1., 0.]

# model parameters
theta0 = Array(Float64, 1)

# dynamics
A = [1. 0.1; 0. 1.]
B = Array(Float64, 2, 1)
B[:, 1] = [0., 1.]
function f(k::Int64, x::Array{Float64, 1}, u::Array{Float64, 1}, theta::Array{Float64, 1})
  return A*x + B*u
end

# reward function
Q = -eye(2)
R = -eye(1)
function g(k::Int64, x::Array{Float64, 1}, u::Array{Float64, 1}, theta::Array{Float64, 1})
    return dot(x, Q*x) + dot(u, R*u)
end

# terminal reward
function g_terminal(x::Array{Float64, 1})
    return dot(x, Q*x)
end

# reduction functions
function rho(x::Array{Float64, 1})
  return x
end

function rho_bar_inverse(x_bar::Array{Float64, 1})
  return x_bar
end

# input grid
ugrid = (linspace(-0.6, 0.6, 101),)

# state grid
xgrid = (linspace(-1, 1, 51), linspace(-1, 1, 51))

function quadratic_cost_gridded(P, x_grid)
  return [dot([x y], P*[x; y]) for x in xgrid[1], y in xgrid[2]]
end

# test that the exact LQR solution for the value function agrees with the gridded DP solution up to a threshold of epsilon
epsilon = 1.0
J = DynamicProgramming.dp_reduced(f, g, g_terminal, rho, rho_bar_inverse, ugrid, xgrid, theta0, N)
J_easy = DynamicProgramming.dp(f, g, g_terminal, ugrid, xgrid, theta0, N)
@test_approx_eq_eps(J, J_easy, epsilon)
F = zeros(length(ugrid), length(xgrid), N)
P = zeros(size(Q)..., N+1)
P[:, :, N+1] = Q
for k = N:-1:1
  F[:, :, k] = inv(R + B'*P[:, :, k+1]*B)*B'*P[:, :, k+1]*A
  P[:, :, k] = A'*P[:, :, k+1]*A - (A'*P[:, :, k+1]*B)*inv(R + B'*P[:, :, k+1]*B)*(B'*P[:, :, k+1]*A) + Q
  # @test_approx_eq_eps(quadratic_cost_gridded(P[:, :, k], xgrid), J[:, :, k], epsilon)
end

# test that a corresponding trajectory rollout agrees up to a threshold epsilon
epsilon = 0.1
u = zeros(length(ugrid), N)
x = zeros(size(x0)..., N+1)
x[:, 1] = x0
for k=1:N
  u[:, k] = -F[:, :, k]*x[:, k]
  x[:, k+1] = A*x[:, k] + B*u[:, k]
end
x_rollout, u_rollout = DynamicProgramming.dp_rollout_reduced(J, x0, f, g, rho, ugrid, xgrid, theta0, N)
x_rollout_easy, u_rollout_easy = DynamicProgramming.dp_rollout(J, x0, f, g, ugrid, xgrid, theta0, N)
@test_approx_eq_eps(x, x_rollout, epsilon)
@test_approx_eq_eps(u, u_rollout, epsilon)
@test_approx_eq_eps(x, x_rollout_easy, epsilon)
@test_approx_eq_eps(u, u_rollout_easy, epsilon)
end
