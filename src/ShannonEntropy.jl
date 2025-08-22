module ShannonEntropy

export ShannonEntropyMethod
export Hist, KSG
export shannon

using StatsBase
using NearestNeighbors
using ..Utils

"""
    ShannonEntropyMethod

An abstract type for different methods of calculating Shannon entropy.
"""
abstract type ShannonEntropyMethod end

"""
    Hist(; bins::Tuple = (-1,))

A method for calculating Shannon entropy using histograms.

# Fields
- `bins::Tuple`: A tuple specifying the binning strategy for the histogram.
                 If `(-1,)`, the binning is determined automatically by `StatsBase.fit`.
                 Otherwise, a tuple of bin edges for each dimension should be provided.
"""
Base.@kwdef struct Hist <: ShannonEntropyMethod
    bins::Tuple = (-1,)
end

"""
    shannon(method::Hist, x::AbstractVector...)

Calculates the Shannon entropy of a set of variables `x` using a histogram-based method.

# Arguments
- `method::Hist`: The histogram-based Shannon entropy calculation method.
- `x::AbstractVector...`: One or more vectors representing the data for which to calculate the entropy. Each vector is a dimension of the data.

# Returns
- `H::Float64`: The calculated Shannon entropy.

# Details
The function first fits a histogram to the data `x`. The probability density function (PDF) is then approximated from the histogram. Finally, the Shannon entropy is calculated by integrating `-p(x) * log(p(x))` over the domain of `x`, where `p(x)` is the PDF.
"""
function shannon(
    method::Hist,
    x::AbstractVector...
)

    n = length(x) # Number of dimensions
    datanum = length(x[1]) # Number of data points
    p = nothing # Histogram object
    H = 0.0 # Shannon entropy

    # Fit a histogram to the data.
    if method.bins == (-1,)
        # If bins are not specified, use automatic binning.
        p = fit(Histogram, x)
    else
        # If bins are specified, check for compatible dimensions.
        (length(method.bins) != length(x)) ? error("Incompatible bin and data dimensions") : nothing
        # Use the specified bins.
        p = fit(Histogram, x, method.bins)
    end

    # Calculate the entropy from the histogram.
    for idx in CartesianIndices(p.weights)
        dx = 1.0 # Volume of the bin
        for i in 1:n
            # Calculate the width of the bin in each dimension.
            dx *= p.edges[i][idx.I[i]+1] - p.edges[i][idx.I[i]]
        end
        # Calculate the probability density in the bin.
        pdf = p.weights[idx] / dx / datanum
        if pdf >= 1.0e-10 # Avoid taking the log of zero.
            # Add the contribution of the bin to the entropy.
            H -= pdf * log(pdf) * dx
        end
    end

    return H

end


"""
    KSG(; k::Int = 5)

A method for calculating Shannon entropy using the Kozachenko-Leonenko (KSG) estimator.

# Fields
- `k::Int`: The number of nearest neighbors to consider for each point.
"""
Base.@kwdef struct KSG <: ShannonEntropyMethod
    k::Int = 5
end

"""
    shannon(method::KSG, x::AbstractVector...)

Calculates the Shannon entropy of a set of variables `x` using the KSG estimator.

# Arguments
- `method::KSG`: The KSG Shannon entropy calculation method.
- `x::AbstractVector...`: One or more vectors representing the data for which to calculate the entropy. Each vector is a dimension of the data.

# Returns
- `H::Float64`: The calculated Shannon entropy.

# Details
The function uses the k-nearest neighbors (k-NN) algorithm to estimate the probability density function (PDF) of the data. The Shannon entropy is then calculated based on the distances to the k-th nearest neighbors. This method is particularly useful for high-dimensional data.
"""
function shannon(
    method::KSG,
    x::AbstractVector...
)

    n = length(x) # Number of dimensions
    datanum = length(x[1]) # Number of data points
    points = transpose(hcat(x...)) # Combine the input vectors into a matrix
    k = method.k # Number of neighbors for KSG
    kdtree = KDTree(points, Chebyshev()) # Build a KDTree for efficient neighbor search
    H = 0.0 # Shannon entropy

    for i in 1:datanum
        # Find the k nearest neighbors for the i-th point.
        point = points[:, i]
        neighbors = knn(kdtree, point, k + 1)
        # Estimate the entropy contribution from the neighbors.
        H += log(neighbors[2][1] * 2.0)
    end
    H = n / datanum * H # Normalize by the number of dimensions
    H += -Utils.digamma(k) + Utils.digamma(datanum)

    return H

end


end
