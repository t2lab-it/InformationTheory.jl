module KLdivergence

export MutualInformationMethod
export Hist, kNN
export kldiv

using StatsBase
using NearestNeighbors
using ..Utils

"""
    KLdivergenceMethod

An abstract type for different methods of calculating KL divergence.
"""
abstract type KLdivergenceMethod end

"""
    Hist(; bins_x::Tuple = (-1,), bins_y::Tuple = (-1,))

A method for calculating KL divergence using histograms.

# Fields
- `bins_x::Tuple`: A tuple specifying the binning strategy for the histogram of the first distribution (P).
                 If `(-1,)`, the binning is determined automatically by `StatsBase.fit`.
                 Otherwise, a tuple of bin edges for each dimension should be provided.
- `bins_y::Tuple`: A tuple specifying the binning strategy for the histogram of the second distribution (Q).
                 If `(-1,)`, the binning is determined automatically by `StatsBase.fit`.
                 Otherwise, a tuple of bin edges for each dimension should be provided.
"""
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

"""
    kldiv(method::Hist, x::Tuple, y::Tuple)

Calculates the KL divergence between two distributions, P and Q, represented by data `x` and `y`, using a histogram-based method.

# Arguments
- `method::Hist`: The histogram-based KL divergence calculation method.
- `x::Tuple`: A tuple of vectors representing the data for the first distribution (P). Each vector is a dimension.
- `y::Tuple`: A tuple of vectors representing the data for the second distribution (Q). Each vector is a dimension.

# Returns
- `D::Float64`: The calculated KL divergence D(P||Q).

# Details
The function fits histograms to the data `x` and `y` to approximate their probability density functions (PDFs), p(x) and q(y). The KL divergence is then calculated by integrating `p(x) * log(p(x) / q(y))` over the domain.
"""
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

"""
    kNN(; k::Int = 5)

A method for calculating KL divergence using a k-nearest neighbors (k-NN) based estimator.

# Fields
- `k::Int`: The number of nearest neighbors to consider for each point.
"""
Base.@kwdef struct kNN <: KLdivergenceMethod
    k::Int = 5
end

"""
    kldiv(method::kNN, x::Tuple, y::Tuple)

Calculates the KL divergence between two distributions, P and Q, represented by data `x` and `y`, using a k-NN based method.

# Arguments
- `method::kNN`: The k-NN based KL divergence calculation method.
- `x::Tuple`: A tuple of vectors representing the data for the first distribution (P).
- `y::Tuple`: A tuple of vectors representing the data for the second distribution (Q).

# Returns
- `D::Float64`: The calculated KL divergence D(P||Q).

# Details
This function uses a non-parametric method to estimate KL divergence based on the distances to the k-nearest neighbors in the data `x` and `y`. It is particularly useful for high-dimensional data.
"""
function kldiv(
    method::kNN,
    x::Tuple,
    y::Tuple
)

    datanum_x = length(x[1]) # Number of data points
    datanum_y = length(y[1]) # Number of data points
    D = 0.0 # KL divergence

    points_x = transpose(hcat(x...))
    points_y = transpose(hcat(y...)) # Combine the input vectors into a matrix
    k = method.k # Number of neighbors for kNN
    kdtree_x = KDTree(points_x, Chebyshev())
    kdtree_y = KDTree(points_y, Chebyshev())

    for i in 1:datanum_x
        # Find the k nearest neighbors for the i-th point.
        point_x = points_x[:, i]
        ρ = knn(kdtree_x, point_x, k + 1)[2][1]
        ν = knn(kdtree_y, point_x, k)[2][1]

        distance = maximum(hcat(ρ, ν))

        Nx = inrangecount(kdtree_x, point_x, distance) - 1
        Ny = inrangecount(kdtree_y, point_x, distance)

        D += Utils.digamma(Nx) - Utils.digamma(Ny)
    end
    D /= datanum_x
    D += log(datanum_y / (datanum_x - 1))

    return D
end

end
