
# TODO: handle multidimensional case

"""
    product_integral(f, g, [, solver])

Compute the integral of `f .* g` over [0,1].
"""
function product_integral(f, g, solver=HCubatureJL())
    function fun(u,_)
        f(u) .* g(u)
    end
    prob = QuadratureProblem{false}(fun, 0, 1, ())
    solve(prob, solver).u
end

"""
    kernel_integral(k, f, t [, solver])

Compute the kernel integral `∫ k(t,s)f(s)ds`.
"""
function kernel_integral(k, f, t, solver=HCubatureJL())
    fun(u) = k(t,u)
    return product_integral(fun, f, solver)
end

"""
    kernel_transpose_integral(k, f, t [, solver])

Compute the kernel integral `∫ k(s,t)f(s)ds`.
"""
function kernel_transpose_integral(k, f, t, solver=HCubatureJL())
    fun(u) = k(u,t)
    return product_integral(fun, f, solver)
end

"""
    kernel_double_integral(k, f, g, [, solver])

Compute the kernel integral `∫∫ k(t,s)f(t)g(s) ds dt`.
"""
function kernel_double_integral(k, f, g, solver=HCubatureJL())
    function fun(u,_)
        f(u[1]) .* k(u[1], u[2]) .* g(u[2])
    end
    prob = QuadratureProblem{false}(fun, [0,0], [1,1], ())
    solve(prob, solver).u
end