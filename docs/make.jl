using SpectralLearning
using Documenter

DocMeta.setdocmeta!(SpectralLearning, :DocTestSetup, :(using SpectralLearning); recursive=true)

makedocs(;
    modules=[SpectralLearning],
    authors="Cédric Simal, University of Namur",
    repo="https://github.com/csimal/SpectralLearning.jl/blob/{commit}{path}#{line}",
    sitename="SpectralLearning.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://csimal.github.io/SpectralLearning.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/csimal/SpectralLearning.jl",
    devbranch="main",
)
