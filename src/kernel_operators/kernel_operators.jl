
"""
    AbstractOperator

An abstract type for linear operators on Banach spaces.
"""
abstract type AbstractOperator end

"""
    KernelOperator{K,S} <: AbstractOperator

A type for representing kernel integral operators over Hilbert spaces.
"""
struct KernelOperator{K,S} <: AbstractOperator
    kernel::K
    solver::S
end

KernelOperator(k) = KernelOperator(k,HCubatureJL())

function eval_(ko::KernelOperator, f; solver=ko.solver)
    k = ko.kernel
    fun = function(u,p) 
        k(p,u) * f(u)
    end
    return function(x)
        prob = QuadratureProblem{false}(fun,0,1,x)
        solve(prob, solver).u
    end
end

function (ko::KernelOperator)(x)
    eval_(ko, x)
end

Base.:*(ko::KernelOperator, x) = ko(x)

function ChainRulesCore.rrule(ko::KernelOperator, x)
    y = ko(x)
    function ko_pullback(δy)
        δko = NoTangent()
        δx = t -> kernel_transpose_integral(ko.kernel, δy, t, ko.solver)
        return δko, δx
    end
    return y, ko_pullback
end

"""
    FourierKernel{T<:Real}

An object representing a Fourier kernel function ``k(x,y) = sin (ω  π xy + θ)``
"""
struct FourierKernel{T<:Real}
    ω::T
    θ::T
end

FourierKernel(; a = 0.0, b = 1.0) = FourierKernel((b-a), a)

(ff::FourierKernel)(x,y) = sin(ff.ω*π*x*y + ff.θ)

"""
    PolynomialKernel{M<:AbstractMatrix}

A type representing a polynomial kernel function.
"""
struct PolynomialKernel{M<:AbstractMatrix}
    m::M
end

function (pk::PolynomialKernel)(x,y)
    (m,n) = size(pk.m)
    X = [x^(k-1) for k in 1:m]
    Y = [y^(k-1) for k in 1:n]
    X * pk.m * Y
end