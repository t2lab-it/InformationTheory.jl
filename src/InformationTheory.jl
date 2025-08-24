module InformationTheory

export shannon, mutual, kldiv, c_mutual
export ShannonEntropy, MutualInformation, KLdivergence, ConditionalMutualInformation

include("Utils.jl")
export Utils

include("ShannonEntropy.jl")
using .ShannonEntropy

include("MutualInformation.jl")
using .MutualInformation

include("KLdivergence.jl")
using .KLdivergence

include("ConditionalMutualInformation.jl")
using .ConditionalMutualInformation

end
