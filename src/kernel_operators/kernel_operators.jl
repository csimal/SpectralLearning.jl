
abstract type AbstractOperator end

struct KernelOperator{K,S} <: AbstractOperator
    kernel::K
    solver::S
end

KernelOperator(k) = KernelOperator(k,QuadGKJL())

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

function (ko::KernelOperator)(f)
    eval_(ko, f)
end

#Base.*(ko::KernelOperator, f::Function) = ko(f)

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