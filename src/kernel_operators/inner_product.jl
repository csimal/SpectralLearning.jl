
"""
    InnerProduct{F,S}

An struct representing a linear form on a Hilbert space as an inner product ``⟨w, x⟩``.
"""
struct InnerProduct{F,S}
    w::F
    solver::S
end

InnerProduct(w) = InnerProduct(w, HCubatureJL())

function inner_product(f, g; solver = HCubatureJL())
    function fun(u,_)
        f(u) .* g(u)
    end
    prob = QuadratureProblem{false}(fun, 0, 1, [])
    solve(prob, solver).u
end

function (ip::InnerProduct)(x)
    inner_product(ip.w, x, solver=ip.solver)
end

Flux.@functor InnerProduct

function ChainRulesCore.rrule(ip::InnerProduct, x)
    y = ip(x)
    function ip_pullback(ȳ)
        ∇w(u) = ForwardDiff.gradient(f -> f(u), w)
        s = inner_product(∇w, x)

        īp = Tangent{InnerProduct}(; w= ȳ .* s, solver=NoTangent())
        x̄ = t -> ȳ * ip.w(t)
        return īp, x̄
    end
    return y, ip_pullback
end

function ChainRulesCore.rrule(ip::InnerProduct{<:ParametricFun,S}, x) where {S}
    y = ip(x)
    function ip_pullback(δy)
        ∇w(t) = grad(ip.w, t)
        s = inner_product(∇w, x)
        δw = from_grad(ip.w, s * δy)

        δip = Tangent{InnerProduct}(; w = δw, solver=NoTangent())
        δx = t -> δy * ip.w(t)
        return δip, δx
    end
    return y, ip_pullback
end