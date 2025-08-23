using InformationTheory
using Test

@testset "ShannonEntropy.jl" begin
    # Write your tests here.
    include("ShannonEntropy.jl")
end

@testset "MutualInformation.jl" begin
    # Write your tests here.
    include("MutualInformation.jl")
end
