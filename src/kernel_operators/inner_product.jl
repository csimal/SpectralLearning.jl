
"""
    InnerProduct{F,S}

An struct representing a linear form on a Hilbert space as an inner product ``⟨w, x⟩``.
"""
struct InnerProduct{F,S}
    w::F
    solver::S
end

InnerProduct(w) = InnerProduct(w, HCubatureJL())

function (ip::InnerProduct)(x)
    product_integral(ip.w, x, ip.solver)
end

Flux.@functor InnerProduct

function w_pullback(ip::InnerProduct{<:ParametricFun,S}, x, δy) where {S}
    ∇w(u) = to_grad(f -> f(u), ip.w)
    s = product_integral(∇w, x)
    return from_grad(ip.w, s * δy)
end

w_pullback(ip, x, δy) = ZeroTangent()

function ChainRulesCore.rrule(ip::InnerProduct, x)
    y = ip(x)
    function ip_pullback(ȳ)
        δw = w_pullback(ip, x, ȳ)
        īp = Tangent{InnerProduct}(; w= δw, solver=NoTangent())
        x̄ = t -> ȳ * ip.w(t)
        return īp, x̄
    end
    return y, ip_pullback
end