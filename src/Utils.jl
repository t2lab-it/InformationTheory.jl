module Utils

export digamma

function digamma(x::Real)
    return log(x) - 1 / (2x)
end

end
