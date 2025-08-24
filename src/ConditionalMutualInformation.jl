module ConditionalMutualInformation

export ConditionalMutualInformationMethod
export Hist, kNN
export c_mutual

using StatsBase
using NearestNeighbors
using ..Utils

"""
    ConditionalMutualInformationMethod

An abstract type for different methods of calculating conditional mutual information.
"""
abstract type ConditionalMutualInformationMethod end

function c_mutual(method::ConditionalMutualInformationMethod, x::AbstractVector, y::AbstractVector, z::AbstractVector)
    return c_mutual(method, (x,), (y,), (z,))
end

function c_mutual(method::ConditionalMutualInformationMethod, x::Tuple, y::AbstractVector, z::AbstractVector)
    return c_mutual(method, x, (y,), (z,))
end

function c_mutual(method::ConditionalMutualInformationMethod, x::AbstractVector, y::Tuple, z::AbstractVector)
    return c_mutual(method, (x,), y, (z,))
end

function c_mutual(method::ConditionalMutualInformationMethod, x::Tuple, y::AbstractVector, z::Tuple)
    return c_mutual(method, x, (y,), z)
end

function c_mutual(method::ConditionalMutualInformationMethod, x::AbstractVector, y::Tuple, z::Tuple)
    return c_mutual(method, (x,), y, z)
end

"""
    Hist(; bins_x::Tuple = (-1,), bins_y::Tuple = (-1,), bins_z::Tuple = (-1,))

A method for calculating conditional mutual information using histograms.

# Fields
- `bins_x::Tuple`: A tuple specifying the binning strategy for the `x` variable.
- `bins_y::Tuple`: A tuple specifying the binning strategy for the `y` variable.
- `bins_z::Tuple`: A tuple specifying the binning strategy for the `z` variable.
  If `(-1,)` for a variable, the binning is determined automatically by `StatsBase.fit`.
  Otherwise, a tuple of bin edges for each dimension should be provided.
"""
Base.@kwdef struct Hist <: ConditionalMutualInformationMethod
    bins_x::Tuple = (-1,)
    bins_y::Tuple = (-1,)
    bins_z::Tuple = (-1,)
end

"""
    c_mutual(method::Hist, x::Tuple, y::Tuple, z::Tuple)

Calculates the conditional mutual information `I(X;Y|Z)` using a histogram-based method.

# Arguments
- `method::Hist`: The histogram-based calculation method.
- `x::Tuple`: A tuple of vectors representing the data for variable X.
- `y::Tuple`: A tuple of vectors representing the data for variable Y.
- `z::Tuple`: A tuple of vectors representing the data for variable Z.

# Returns
- `I::Float64`: The calculated conditional mutual information.

# Details
The function first fits a histogram to the joint data `(x, y, z)`. The probability density function (PDF) is then approximated from the histogram. Finally, the conditional mutual information is calculated from the joint and marginal probabilities.
The formula used is:
`I(X;Y|Z) = sum(p(x,y,z) * log(p(x,y,z) * p(z) / (p(x,z) * p(y,z))))`
"""
function c_mutual(
    method::Hist,
    x::Tuple,
    y::Tuple,
    z::Tuple
)

    nx = length(x) # Number of dimensions x
    ny = length(y) # Number of dimensions y
    nz = length(z) # Number of dimensions z
    datanum = length(x[1]) # Number of data points
    pxz = nothing # Histogram object p(x,z)
    pyz = nothing # Histogram object p(y,z)
    pxyz = nothing # Histogram object p(x,y,z)
    pz = nothing # Histogram object p(z)
    I = 0.0 # Mutual information

    # Fit a histogram to the data.
    if method.bins_x == (-1,) || method.bins_y == (-1,)
        # If bins are not specified, use automatic binning.
        pxyz = fit(Histogram, (x..., y..., z...))
        pxz = fit(Histogram, (x..., z...), (pxyz.edges[1:nx]..., pxyz.edges[nx+ny+1:end]...))
        pyz = fit(Histogram, (y..., z...), (pxyz.edges[nx+1:nx+ny]..., pxyz.edges[nx+ny+1:end]...))
        pz = fit(Histogram, z, pxyz.edges[nx+ny+1:end])
    else
        # If bins are specified, check for compatible dimensions.
        (length(method.bins_x) != length(x)) ? error("Incompatible x bin and data dimensions") : nothing
        (length(method.bins_y) != length(y)) ? error("Incompatible y bin and data dimensions") : nothing
        # Use the specified bins.
        pxyz = fit(Histogram, (x..., y..., z...), (method.bins_x..., method.bins_y..., method.bins_z...))
        pxz = fit(Histogram, (x..., z...), (method.bins_x..., method.bins_z...))
        pyz = fit(Histogram, (y..., z...), (method.bins_y..., method.bins_z...))
        pz = fit(Histogram, z, method.bins_z)
    end

    # Calculate the entropy from the histogram.
    for idx in CartesianIndices(pxyz.weights)
        dx = 1.0 # Volume of the bin (x-direction)
        dy = 1.0 # Volume of the bin (y-direction)
        dz = 1.0 # Volume of the bin (z-direction)
        for i in 1:nx
            # Calculate the width of the bin in each dimension.
            dx *= pxyz.edges[i][idx.I[i]+1] - pxyz.edges[i][idx.I[i]]
        end
        for i in (nx+1):(nx+ny)
            # Calculate the width of the bin in each dimension.
            dy *= pxyz.edges[i][idx.I[i]+1] - pxyz.edges[i][idx.I[i]]
        end
        for i in (nx+ny+1):(nx+ny+nz)
            # Calculate the width of the bin in each dimension.
            dz *= pxyz.edges[i][idx.I[i]+1] - pxyz.edges[i][idx.I[i]]
        end
        # Calculate the probability density in the bin.
        pdf_xyz = pxyz.weights[idx] / (dx * dy * dz) / datanum
        pdf_xz = pxz.weights[CartesianIndex(idx.I[1:nx]..., idx.I[(nx+ny+1):end]...)] / (dx * dz) / datanum
        pdf_yz = pyz.weights[CartesianIndex(idx.I[(nx+1):(nx+ny)]..., idx.I[(nx+ny+1):end]...)] / (dy * dz) / datanum
        pdf_z = pz.weights[CartesianIndex(idx.I[(nx+ny+1):end])] / (dz * datanum)
        if pdf_xyz >= 1.0e-10 # Avoid taking the log of zero.
            # Add the contribution of the bin to the entropy.
            I += pdf_xyz * log((pdf_xyz * pdf_z) / (pdf_xz * pdf_yz)) * dx * dy * dz
        end
    end

    return I

end

"""
    kNN(; k::Int = 5)

A method for calculating conditional mutual information using the k-Nearest Neighbors (k-NN) estimator.

# Fields
- `k::Int`: The number of nearest neighbors to consider for each point.
"""
Base.@kwdef struct kNN <: ConditionalMutualInformationMethod
    k::Int = 5
end

"""
    c_mutual(method::kNN, x::Tuple, y::Tuple, z::Tuple)

Calculates the conditional mutual information `I(X;Y|Z)` using a k-NN based method.

# Arguments
- `method::kNN`: The k-NN based calculation method.
- `x::Tuple`: A tuple of vectors representing the data for variable X.
- `y::Tuple`: A tuple of vectors representing the data for variable Y.
- `z::Tuple`: A tuple of vectors representing the data for variable Z.

# Returns
- `I::Float64`: The calculated conditional mutual information.

# Details
This function uses the k-nearest neighbors (k-NN) algorithm to estimate the conditional mutual information. The estimation is based on the number of neighbors of each point within a certain distance in different subspaces (Z, XZ, YZ). This method is particularly useful for high-dimensional data where histogram-based methods fail.
"""
function c_mutual(
    method::kNN,
    x::Tuple,
    y::Tuple,
    z::Tuple
)

    nx = length(x) # Number of dimensions x
    ny = length(y) # Number of dimensions y
    nz = length(z) # Number of dimensions z
    datanum = length(x[1]) # Number of data points
    I = 0.0 # Mutual information

    points_x = transpose(hcat(x...)) # Combine the input vectors into a matrix
    points_y = transpose(hcat(y...)) # Combine the input vectors into a matrix
    points_z = transpose(hcat(z...)) # Combine the input vectors into a matrix
    k = method.k # Number of neighbors for kNN

    # Build a KDTree for efficient neighbor search
    kdtree_xyz = KDTree(vcat(points_x, vcat(points_y, points_z)), Chebyshev())
    kdtree_xz = KDTree(vcat(points_x, points_z), Chebyshev())
    kdtree_yz = KDTree(vcat(points_y, points_z), Chebyshev())
    kdtree_z = KDTree(points_z, Chebyshev())

    for i in 1:datanum
        # Find the k nearest neighbors for the i-th point.
        point_x = points_x[:, i]
        point_y = points_y[:, i]
        point_z = points_z[:, i]
        point_xyz = vcat(point_x, vcat(point_y, point_z))
        neighbors_xyz = knn(kdtree_xyz, point_xyz, k + 1)

        Nz = inrangecount(kdtree_z, point_z, neighbors_xyz[2][1]) - 1
        Nxz = inrangecount(kdtree_xz, vcat(point_x, point_z), neighbors_xyz[2][1]) - 1
        Nyz = inrangecount(kdtree_yz, vcat(point_y, point_z), neighbors_xyz[2][1]) - 1

        I += Utils.digamma(Nz + 1) - Utils.digamma(Nxz + 1) - Utils.digamma(Nyz + 1)
    end
    I /= datanum
    I += Utils.digamma(k)

    return I
end

end
