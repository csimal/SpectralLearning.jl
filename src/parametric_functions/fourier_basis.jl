
"""
    Fourier

A functor struct representing the Fourier basis.
"""
struct Fourier end

"""
    fourier_basis(k,x)

Evaluate the `k`-th Fourier basis function at `x`.
"""
function fourier_basis(k,x)
    cos((div(k,2)*x - iseven(k)/2)*π)
end

fourier_basis(k,x::AbstractVector) = fourier_basis.(k,x)

(::Fourier)(k,x) = fourier_basis(k,x)