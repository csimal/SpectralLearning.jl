
"""
    SpectralNeuralOperator{F,L,F1,F2,P}

A type representing a Neural Operator whose forward pass is of the form

    y(t) = σ((B*x)(t) - λ(t)*(B*x)(t) + b(t))

where `x`, `λ` and `b` are functions, and `B` is an integral kernel operator. 

By default, `y` is projected in the Chebyshev polynomial basis before outputing. This is done to avoid computing multiple nested integrals when stacking multiple layers.
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

function (sno::SpectralNeuralOperator)(x)
    B, λ, b, σ = sno.B, sno.λ, sno.b, sno.σ
    y = B(x)
    z = t -> let yt = y(t)
        σ(yt - λ(t)*yt + b(t))
    end
    sno.project(z)
end

function ChainRulesCore.rrule(sno::SpectralNeuralOperator, x)
    y = sno(x)
    function sno_pullback(δy)
        δB = NoTangent()
        δσ = NoTangent()
        ∇b(t) = grad(b, t)
    end
    return y, sk_pullback
end
