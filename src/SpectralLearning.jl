module SpectralLearning

using Quadrature
using HCubature
using ApproxFun
using Flux
using ChainRulesCore
using Zygote
using ForwardDiff

include("spectral_layer.jl")

export Spectral, spectralize

include("utils/integrals.jl")

include("parametric_functions/ParametricFunctions.jl")
using .ParametricFunctions

include("kernel_operators/inner_product.jl")
include("kernel_operators/kernel_operators.jl")

export InnerProduct
export AbstractOperator, KernelOperator
export FourierKernel

include("neural_operators/spectral_neural_operator.jl")
include("neural_operators/finite_dim_operator.jl")

export SpectralNeuralOperator
export FiniteDimOperator

end
