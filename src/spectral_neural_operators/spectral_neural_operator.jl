
"""
    SpectralNeuralOperator{F,L,F1,F2,P}

A type representing a Neural Operator whose forward pass is of the form

    y(t) = σ((B*x)(t) - λ(t)*(B*x)(t) + b(t))

where `x`, `λ` and `b` are functions, and `B` is an integral kernel operator. 

By default, `y` is projected in the Chebyshev polynomial basis before outputing. This is done to avoid computing multiple nested integrals when stacking multiple layers.

## Fields
 * `B::L` A linear operator on functions ``[0,1] → R``, generally a kernel operator.
 * `λ::F1` a function ``[0,1] → R``
 * `b::F2` a function ``[0,1] → R`` used as bias
 * `σ::F` the activation function
 * `project::P` a projection operator used on the output of the layer. Since by default, the output is a lazy function, this is meant to 
"""
struct SpectralNeuralOperator{F,L,F1,F2,P}
    B::L # Linear Operator
    λ::F1 # Spectral Scaling Function
    b::F2 # Bias function
    σ::F # activation function
    project::P # Projection operator
end

SpectralNeuralOperator(B, λ, b) = SpectralNeuralOperator(B, λ, b, identity, f -> Fun(f, 0..1))

Flux.@functor SpectralNeuralOperator

function kernel_fun(sno::SpectralNeuralOperator{F,<:KernelOperator,F1,F2,P}) where {F1,F2,F,P}
    return function k_fun(t,s)
        (1 - sno.λ(t))*sno.B.kernel(t,s)
    end
end

function (sno::SpectralNeuralOperator)(x)
    B, λ, b, σ = sno.B, sno.λ, sno.b, sno.σ
    y = B(x)
    z = t -> let yt = y(t)
        σ(yt - λ(t)*yt + b(t))
    end
    sno.project(z)
end

function b_pullback(sno::SpectralNeuralOperator{F,L,F1,F2,P}, δz, y) where {F,L,F1,F2<:ParametricFun,P}
    dσ(u) = ForwardDiff.derivative(sno.σ, u)
    function fun(u)
        dσ(y(u)) .* to_grad(sno.b, u)
    end
    δb = product_integral(fun, δz)
    return from_grad(sno.b, δb)
end

# default behaviour, no tangent
b_pullback(sno, δz, δy) = ZeroTangent()

function λ_pullback(sno::SpectralNeuralOperator{F,L,F1,F2,P}, δz, y) where {F,L,F1<:ParametricFun,F2,P}
    dσ(u) = ForwardDiff.derivative(sno.σ, u)
    function fun(u)
        dσ(y(u)) .* to_grad(sno.λ, u)
    end
    δλ =  kernel_double_integral(kernel_fun(sno), fun, δz)
    return from_grad(sno.λ, δλ)
end

λ_pullback(sno, δz, δy) = ZeroTangent()

function x_pullback(sno::SpectralNeuralOperator, δz, y)
    dσ(u) = ForwardDiff.derivative(sno.σ, u)
    function δx(s)
        dσ(y(s)) * kernel_transpose_integral(kernel_fun(sno), δz, s)
    end
    return sno.project(δx)
end

function ChainRulesCore.rrule(sno::SpectralNeuralOperator, x)
    Bx = sno.B(x)
    y = sno.project(t -> (1 - sno.λ(t))*Bx(t) + sno.b(t))
    z = t -> sno.σ( (1 - sno.λ(t)) * y(t) + sno.b(t) )
    function sno_pullback(δz)
        δB = ZeroTangent()
        δλ = λ_pullback(sno, δz, y)
        δb = b_pullback(sno, δz, y)
        δσ = NoTangent()
        δproject = NoTangent()
        δx = x_pullback(sno, δz, y)
        return Tangent{SpectralNeuralOperator}(;B=δB, λ=δλ, b=δb, σ=δσ, project=δproject), δx
    end
    return sno.project(z), sno_pullback
end
