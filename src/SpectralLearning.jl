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

export InnerProduct

include("kernel_operators/kernel_operators.jl")

export AbstractOperator, KernelOperator
export FourierKernel

include("spectral_neural_operators/spectral_neural_operator.jl")

export SpectralNeuralOperator

end
