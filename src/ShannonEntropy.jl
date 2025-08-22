module ShannonEntropy

export ShannonEntropyMethod
export Hist
export shannon

using StatsBase
using ..Utils

abstract type ShannonEntropyMethod end

Base.@kwdef struct Hist <: ShannonEntropyMethod
    bins::Tuple = (-1,)
end

function shannon(
    method::Hist,
    x::AbstractVector...
)

    n = length(x)
    datanum = length(x[1])
    p = nothing
    H = 0.0

    if method.bins == (-1,)
        p = fit(Histogram, x)
    else
        (length(method.bins) != length(x)) ? error("Incompatible bin and data dimensions") : nothing
        p = fit(Histogram, x, method.bins)
    end

    for idx in CartesianIndices(p.weights)
        dx = 1.0
        for i in 1:n
            dx *= p.edges[i][idx.I[i] + 1] - p.edges[i][idx.I[i]]
        end
        pdf = p.weights[idx] / dx / datanum
        if pdf >= 1.0e-10
            H -= pdf * log(pdf) * dx
        end
    end

    return H

end

end
