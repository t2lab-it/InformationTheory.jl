using InformationTheory
using Documenter

DocMeta.setdocmeta!(InformationTheory, :DocTestSetup, :(using InformationTheory); recursive=true)

makedocs(;
    modules=[InformationTheory],
    authors="t2lab-it",
    sitename="InformationTheory.jl",
    format=Documenter.HTML(;
        canonical="https://t2lab-it.github.io/InformationTheory.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => [
            "Shannon Entropy" => "API/ShannonEntropy.md",
            "Mutual Information" => "API/MutualInformation.md",
            "KL Divergence" => "API/KLdivergence.md",
            "Conditional Mutual Information" => "API/ConditionalMutualInformation.md",
        ],
        "Tutorial" => [
            "Shannon Entropy" => "Tutorial/ShannonEntropy.md",
            "Mutual Information" => "Tutorial/MutualInformation.md",
            "KL Divergence" => "Tutorial/KLdivergence.md",
            "Conditional Mutual Information" => "Tutorial/ConditionalMutualInformation.md",
        ]
    ],
)

deploydocs(;
    repo="github.com/t2lab-it/InformationTheory.jl",
    devbranch="main",
)
