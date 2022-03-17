module ParametricFunctions

using ForwardDiff
using ChainRulesCore

export ParametricFun, BasisFun
export Chebyshev, Fourier

"""
    An abstract type representing a parametric function.
"""
abstract type ParametricFun end


"""
    grad(f::ParametricFun, x)

Return the gradient of `f` relative to its parameters at `x`.
"""
function grad end

"""
        from_grad(f::ParametricFun, ∇f)

Create a new instance of the type of `f` from a gradient over its parameters.
"""
function from_grad end

"""
    params(f::ParametricFun)

Return the parameters of `f`.
"""
function params end

include("basis_fun.jl")
include("chebyshev_basis.jl")
include("fourier_basis.jl")


end