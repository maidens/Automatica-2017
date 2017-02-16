module RunTests

using Base.Test

# Test DynamicProgramming.jl to ensure that it reproduces the linear quadratic regulator
include("linear_quadratic.jl")
include("linear_quadratic_symmetry.jl")

println("All tests passed.")

end
