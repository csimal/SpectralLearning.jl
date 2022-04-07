
"""
    FiniteDimOperator{F,W,B}

A type representing a map from [0,1] to ``Rⁿ``, that is

    z[i] = σ(w[i](x) + b[i])

where `w` is a vector of linear forms on the Hilbert space of ``[0,1] → R``-functions.
"""
struct FiniteDimOperator{F,W,B}
    w::Vector{W} # linear forms
    b::Vector{B} # biases
    σ::F # activation function
end

#=
function FiniteDimOperator(out::Integer, σ=identity; init=Flux.glorot_uniform)
    b = init(out)
    FiniteDimOperator(w,b,σ)
end
=#

function Base.show(io::IO, ::MIME"text/plain", x::FiniteDimOperator{F,W,B}) where {F,W,B}
    println(io, "$(length(x.b))-dimensional FiniteDimOperator with\n functional type $W\n bias type $B\n activation function $F")
end

Flux.@functor FiniteDimOperator

function (fdo::FiniteDimOperator)(x)
    w, b = fdo.w, fdo.b
    fdo.σ.([w[i](x) + b[i] for i in 1:length(b)])
end