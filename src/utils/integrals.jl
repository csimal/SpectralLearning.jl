
function product_integral(f, g, solver=HCubatureJL())
    function fun(u,_)
        f(u) .* g(u)
    end
    prob = QuadratureProblem{false}(fun, 0, 1, [])
    solve(prob, solver).u
end