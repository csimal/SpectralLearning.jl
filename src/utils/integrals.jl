
# TODO: handle multidimensional case

function product_integral(f, g, solver=HCubatureJL())
    function fun(u,_)
        f(u) .* g(u)
    end
    prob = QuadratureProblem{false}(fun, 0, 1, ())
    solve(prob, solver).u
end

function kernel_integral(k, f, t, solver=HCubatureJL())
    fun(u) = k(t,u)
    return product_integral(fun, f, solver)
end

function kernel_double_integral(k, f, g, solver=HCubatureJL())
    function fun(u,_)
        f(u[1]) .* k(u[1], u[2]) .* g(u[2])
    end
    prob = QuadratureProblem{false}(fun, [0,0], [1,1], ())
    solve(prob, solver).u
end