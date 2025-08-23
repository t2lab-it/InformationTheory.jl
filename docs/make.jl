using InformationTheory
using Documenter

DocMeta.setdocmeta!(InformationTheory, :DocTestSetup, :(using InformationTheory); recursive=true)

makedocs(;
    modules=[InformationTheory],
    authors="tkrhsmt <tkrhsmt@gmail.com>",
    sitename="InformationTheory.jl",
    format=Documenter.HTML(;
        canonical="https://tkrhsmt.github.io/InformationTheory.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => [
            "Shannon Entropy" => "API/ShannonEntropy.md",
            "Mutual Information" => "API/MutualInformation.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/tkrhsmt/InformationTheory.jl",
    devbranch="main",
)
