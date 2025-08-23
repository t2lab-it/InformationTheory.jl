using Distributions: MvNormal
using StateSpaceSets: StateSpaceSet

N = 10000
c = 0.9
Σ = [1 c; c 1]
N2 = MvNormal([0, 0], Σ)

D2 = StateSpaceSet([rand(N2) for i = 1:N])
x_data = [x[1] for x in D2[:, 1] |> StateSpaceSet]
y_data = [x[1] for x in D2[:, 2] |> StateSpaceSet]
z_data = randn(N)

MI_exact = -0.5 * log(Σ[1, 1] * Σ[2, 2] - Σ[1, 2] * Σ[2, 1])

println("---------- histogram method ----------")

# Estimate mutual information using a histogram
MI_hist = InformationTheory.mutual(InformationTheory.MutualInformation.Hist(), x_data, y_data)
println("Exact MI(X;Y): $MI_exact\t Histogram MI(X;Y): $MI_hist\t error: $(abs(MI_hist - MI_exact))")
@test MI_hist ≈ MI_exact rtol = 0.1

# Estimate joint mutual information using a histogram
MI_joint_hist = InformationTheory.mutual(InformationTheory.MutualInformation.Hist(), x_data, (y_data, z_data))
println("Exact MI(X;Y,Z): $MI_exact\t Histogram MI(X;Y,Z): $MI_joint_hist error: $(abs(MI_joint_hist - MI_exact))")
@test MI_joint_hist ≈ MI_exact rtol = 0.1

# Estimate mutual information using a histogram
MI_hist = InformationTheory.mutual(InformationTheory.MutualInformation.Hist(bins_x=(-4:0.25:4,), bins_y=(-5:0.25:5,)), x_data, y_data)
println("Exact MI(X;Y): $MI_exact\t Histogram MI(X;Y): $MI_hist\t error: $(abs(MI_hist - MI_exact))")
@test MI_hist ≈ MI_exact rtol = 0.1
