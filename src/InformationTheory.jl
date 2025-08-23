module InformationTheory

include("Utils.jl")
using .Utils

include("ShannonEntropy.jl")
using .ShannonEntropy

include("MutualInformation.jl")
using .MutualInformation

include("KLdivergence.jl")
using .KLdivergence

include("ConditionalMutualInformation.jl")
using .ConditionalMutualInformation

end
