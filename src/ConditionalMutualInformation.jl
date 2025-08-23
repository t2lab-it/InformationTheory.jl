module ConditionalMutualInformation

export ConditionalMutualInformationMethod
export Hist#, kNN
export c_mutual

using StatsBase
using NearestNeighbors
using ..Utils

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

Base.@kwdef struct Hist <: ConditionalMutualInformationMethod
    bins_x::Tuple = (-1,)
    bins_y::Tuple = (-1,)
    bins_z::Tuple = (-1,)
end

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

end
