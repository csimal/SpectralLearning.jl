
"""
    Spectral{F,M,V1,V2,V3}

Create a spectral layer, whose forward pass is given by

    σ.(B*(λ_in.*x) .- λ_out.*(B*x) .+ b)

where `B` is a static matrix, and `λ_in`, `λ_out` and `b` are trainable vectors of parameters.

These layers can be used as drop-in replacement to traditional dense layers, and are usually as good as dense layers for similar amounts of parameters (but spectral layers are wider for the same number of parameters).

# Examples
```jldoctest
julia> sp = Spectral(5, 2)
Spectral(5 => 2)
```
"""
struct Spectral{F, M, V1, V2, V3}
    B::M
    λ_in::V1
    λ_out::V2
    b::V3
    σ::F
end

function Spectral(in::Integer, out::Integer, σ=identity;
    init=Flux.glorot_uniform,
    bias=true
    )
    B = init(out, in)
    λ_in = init(in)
    λ_out = init(out)
    b = bias ? init(out) : bias
    Spectral(B, λ_in, λ_out, b, σ)
end

Flux.trainable(s::Spectral) = (s.λ_in, s.λ_out, s.b,)

Flux.@functor Spectral

function (s::Spectral)(x::AbstractVecOrMat)
    B, λ_in, λ_out, b, σ = s.B, s.λ_in, s.λ_out, s.b, s.σ
    σ.(B*(λ_in.*x) .- λ_out.*(B*x) .+ b)
end

function Base.show(io::IO, sp::Spectral)
    print(io, "Spectral(", size(sp.B, 2), " => ", size(sp.B, 1))
    sp.σ == identity || print(io, ", ", sp.σ)
    sp.b == false && print(io, "; bias=false")
    print(io, ")")
end

"""
    spectralize(d::Dense)

Create a spectral layer with the same dimensions as `d`.
"""
function spectralize(d::Dense)
    in, out = size(d.weight)
    Spectral(in, out, d.σ)
end
