
"""
    BasisFun{T,F} <: ParametricFun

A type representing a linear combination of basis functions.
"""
struct BasisFun{T<:AbstractVector{<:Real},F} <: ParametricFun
    a::T
    basis::F
end

_showstr(x::BasisFun{T,F}) where {T,F} = "$(length(x.a))-element BasisFun{\n$T,\n$F}\n$(x.a)"

#Base.show(io::IO, x::BasisFun{T,F}) where {T,F} = _showstr(x)
function Base.show(io::IO, ::MIME"text/plain", x::BasisFun{T,F}) where {T,F}
    println(io, "$(length(x.a))-element BasisFun with basis $F and coefficients")
    println(io, x.a)
end
#Base.show(io::IO, ::MIME, x::BasisFun) = _showstr(x)

"""
    eval_fun(x, a, basis)

Evaluate a linear combination of basis functions at a point `x`.

## Arguments
* `x`: the point at which to evaluate the function
* `a`: a vector of coefficients representing the function
* `basis`: a function of the form `b(k,x)` that evaluates the `k`-th basis element at `x`.
"""
function eval_fun(x, a, basis)
    y = zero(x * a[1])
    for k in 1:length(a)
        y += a[k] * basis(k, x)
    end
    return y
end

function ChainRulesCore.rrule(::typeof(eval_fun), x, a, basis)
    y = eval_fun(x, a, basis)
    function eval_fun_pullback(δy)
        δf = NoTangent()
        δx = @thunk ForwardDiff.derivative(t -> eval_fun(t, a, basis), x) * δy
        δa = [basis(k,x) for k in 1:length(a)] * δy
        return δf, δx, δa, NoTangent()
    end
    return y, eval_fun_pullback
end

function (f::BasisFun)(x)
    eval_fun(x, f.a, f.basis)
end

function grad(f::BasisFun, x)
    [f.basis(k,x) for k in 1:length(f.a)]
end

function from_grad(f::BasisFun, ∇f)
    return BasisFun(∇f, f.basis)
end

params(f::BasisFun) = f.a

Base.:+(f::BasisFun{T,F}, g::BasisFun{T,F}) where {T,F} = BasisFun{T,F}(f.a+g.a, f.basis)
Base.:-(f::BasisFun{T,F}, g::BasisFun{T,F}) where {T,F} = BasisFun{T,F}(f.a-g.a, f.basis)
Base.zero(f::BasisFun) = BasisFun(zero(f.a), f.basis)

Base.length(f::BasisFun) = length(f.a)

Base.:*(α::Number, f::BasisFun) = BasisFun(α * f.a, f.basis)