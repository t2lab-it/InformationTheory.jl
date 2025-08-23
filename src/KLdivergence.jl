module KLdivergence

export MutualInformationMethod
export Hist#, KSG
export kldiv

using StatsBase
using NearestNeighbors
using ..Utils

abstract type KLdivergenceMethod end

Base.@kwdef struct Hist <: KLdivergenceMethod
    bins_x::Tuple = (-1,)
    bins_y::Tuple = (-1,)
end

function kldiv(
    method::KLdivergenceMethod,
    x::AbstractVector,
    y::AbstractVector
)
    return kldiv(method, (x,), (y,))
end

function kldiv(
    method::KLdivergenceMethod,
    x::Tuple,
    y::AbstractVector
)
    return kldiv(method, x, (y,))
end

function kldiv(
    method::KLdivergenceMethod,
    x::AbstractVector,
    y::Tuple
)
    return kldiv(method, (x,), y)
end

function kldiv(
    method::Hist,
    x::Tuple,
    y::Tuple
)

    n = length(x)# Number of dimensions
    (length(y) != n) ? error("Incompatible x and y dimensions") : nothing
    datanum = length(x[1]) # Number of data points
    px = nothing
    py = nothing # Histogram object p(x) and p(y)
    D = 0.0 # KL divergence

    # Fit a histogram to the data.
    if method.bins_x == (-1,) || method.bins_y == (-1,)
        # If bins are not specified, use automatic binning.
        px = fit(Histogram, x)
        py = fit(Histogram, y, px.edges)
    else
        # If bins are specified, check for compatible dimensions.
        (length(method.bins_x) != length(x)) ? error("Incompatible x bin and data dimensions") : nothing
        (length(method.bins_y) != length(y)) ? error("Incompatible y bin and data dimensions") : nothing
        # Use the specified bins.
        px = fit(Histogram, x, method.bins_x)
        py = fit(Histogram, y, method.bins_y)
    end

    # Calculate the entropy from the histogram.
    for idx in CartesianIndices(px.weights)
        dx = 1.0 # Volume of the bin (x-direction)
        for i in 1:n
            # Calculate the width of the bin in each dimension.
            dx *= px.edges[i][idx.I[i]+1] - px.edges[i][idx.I[i]]
        end
        # Calculate the probability density in the bin.
        pdf_x = px.weights[idx] / (dx * datanum)
        pdf_y = py.weights[idx] / (dx * datanum)
        if pdf_x >= 1.0e-10 && pdf_y >= 1.0e-10 # Avoid taking the log of zero.
            # Add the contribution of the bin to the entropy.
            D += pdf_x * log(pdf_x / pdf_y) * dx
        end
    end

    return D

end

end
