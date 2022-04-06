module ParametricFunctions

using ForwardDiff
using ChainRulesCore

include("basis_fun.jl")
include("chebyshev_basis.jl")
include("fourier_basis.jl")

export ParametricFun, BasisFun
export Chebyshev, Fourier

export Parametric, NonParametric

export to_grad, from_grad, params

"""
    ParametricFun

An abstract type representing a parametric function.
"""
abstract type ParametricFun end

"""
    Parametric

A Trait type denoting functions with trainable parameters.

Functions implementing this trait must have methods for `has_params`, `to_grad`, `from_grad` and `params`.
"""
struct Parametric end

"""
    NonParametric

A trait type denoting functions with no parameters.
"""
struct NonParametric end

"""
    has_params(f)

Whether a functor type has trainable parameters.

Override this to return `Parametric()` for your type to declare it as parametric functions.
"""
has_params(f) = NonParametric()
has_params(::ParametricFun) = Parametric()

"""
    to_grad(f, x)

Return the gradient of `f` relative to its parameters at `x`.

The returned object must be an array (or any object that can be used in  a `QuadratureProblem`).
"""
function to_grad end

"""
        from_grad(f, ∇f)

Create a new instance of the type of `f` from a gradient over its parameters.
"""
function from_grad end

"""
    params(f)

Return the parameters of `f`.
"""
function params end

end