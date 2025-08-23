module MutualInformation

export MutualInformationMethod
export Hist, KSG
export mutual

using StatsBase
using NearestNeighbors
using ..Utils

"""
    MutualInformationMethod

An abstract type for different methods of calculating mutual information.
"""
abstract type MutualInformationMethod end

function mutual(
    method::MutualInformationMethod,
    x::AbstractVector,
    y::AbstractVector
)
    return mutual(method, (x,), (y,))
end

function mutual(
    method::MutualInformationMethod,
    x::Tuple,
    y::AbstractVector
)
    return mutual(method, x, (y,))
end

function mutual(
    method::MutualInformationMethod,
    x::AbstractVector,
    y::Tuple
)
    return mutual(method, (x,), y)
end

Base.@kwdef struct Hist <: MutualInformationMethod
    bins_x::Tuple = (-1,)
    bins_y::Tuple = (-1,)
end

function mutual(
    method::Hist,
    x::Tuple,
    y::Tuple
)

    nx = length(x)
    ny = length(y) # Number of dimensions x and y
    datanum = length(x[1]) # Number of data points
    pxy = nothing # Histogram object p(x,y)
    px = nothing
    py = nothing # Histogram object p(x) and p(y)
    I = 0.0 # Mutual information

    # Fit a histogram to the data.
    if method.bins_x == (-1,) || method.bins_y == (-1,)
        # If bins are not specified, use automatic binning.
        pxy = fit(Histogram, (x..., y...))
        px = fit(Histogram, x, pxy.edges[1:nx])
        py = fit(Histogram, y, pxy.edges[nx+1:end])
    else
        # If bins are specified, check for compatible dimensions.
        (length(method.bins_x) != length(x)) ? error("Incompatible x bin and data dimensions") : nothing
        (length(method.bins_y) != length(y)) ? error("Incompatible y bin and data dimensions") : nothing
        # Use the specified bins.
        pxy = fit(Histogram, (x..., y...), (method.bins_x..., method.bins_y...))
        px = fit(Histogram, x, method.bins_x)
        py = fit(Histogram, y, method.bins_y)
    end

    # Calculate the entropy from the histogram.
    for idx in CartesianIndices(pxy.weights)
        dx = 1.0 # Volume of the bin (x-direction)
        dy = 1.0 # Volume of the bin (y-direction)
        for i in 1:nx
            # Calculate the width of the bin in each dimension.
            dx *= pxy.edges[i][idx.I[i]+1] - pxy.edges[i][idx.I[i]]
        end
        for i in (nx+1):(nx+ny)
            # Calculate the width of the bin in each dimension.
            dy *= pxy.edges[i][idx.I[i]+1] - pxy.edges[i][idx.I[i]]
        end
        # Calculate the probability density in the bin.
        pdf_xy = pxy.weights[idx] / (dx * dy) / datanum
        pdf_x = px.weights[CartesianIndex(idx.I[1:nx])] / (dx * datanum)
        pdf_y = py.weights[CartesianIndex(idx.I[(nx+1):end])] / (dy * datanum)
        if pdf_xy >= 1.0e-10 # Avoid taking the log of zero.
            # Add the contribution of the bin to the entropy.
            I += pdf_xy * log(pdf_xy / (pdf_x * pdf_y)) * dx * dy
        end
    end

    return I

end

Base.@kwdef struct KSG <: MutualInformationMethod
    k::Int = 5
end

function mutual(
    method::KSG,
    x::Tuple,
    y::Tuple
)

    nx = length(x)
    ny = length(y) # Number of dimensions x and y
    datanum = length(x[1]) # Number of data points
    I = 0.0 # Mutual information

    points_x = transpose(hcat(x...))
    points_y = transpose(hcat(y...)) # Combine the input vectors into a matrix
    k = method.k # Number of neighbors for KSG
    kdtree_x = KDTree(points_x, Chebyshev())
    kdtree_y = KDTree(points_y, Chebyshev())
    kdtree_xy = KDTree(vcat(points_x, points_y), Chebyshev()) # Build a KDTree for efficient neighbor search

    for i in 1:datanum
        # Find the k nearest neighbors for the i-th point.
        point_x = points_x[:, i]
        point_y = points_y[:, i]
        point_xy = vcat(point_x, point_y)
        neighbors_xy = knn(kdtree_xy, point_xy, k + 1)

        Nx = inrangecount(kdtree_x, point_x, neighbors_xy[2][1]) - 1
        Ny = inrangecount(kdtree_y, point_y, neighbors_xy[2][1]) - 1
        I -= Utils.digamma(Nx + 1) + Utils.digamma(Ny + 1)
    end
    I /= datanum
    I += Utils.digamma(datanum) + Utils.digamma(k)

    return I
end

end
