N = 100000
a, b = 0.8, 0.6
r₁ = 0.7
r₂ = 0.3

z_data  = randn(N)
Ux = randn(N); Uy = randn(N)

epsx = Ux
epsy₁ = r₁ .* Ux .+ sqrt(1 - r₁^2) .* Uy
epsy₂ = r₂ .* Ux .+ sqrt(1 - r₂^2) .* Uy

x_data = a .* z_data .+ sqrt(1 - a^2) .* epsx
y_data₁ = b .* z_data .+ sqrt(1 - b^2) .* epsy₁
y_data₂ = b .* z_data .+ sqrt(1 - b^2) .* epsy₂

CMI_exact_1 = -0.5 * log(1 - r₁^2)
CMI_exact_2 = -0.5 * log(1 - r₂^2)

println("---------- histogram method ----------")

# Estimate mutual information using a histogram
CMI_hist = InformationTheory.c_mutual(InformationTheory.ConditionalMutualInformation.Hist(), x_data, y_data₁, z_data)
println("Exact CMI(X;Y|Z): $CMI_exact_1\t Histogram CMI(X;Y|Z): $CMI_hist\t error: $(abs(CMI_hist - CMI_exact_1))")
@test CMI_hist ≈ CMI_exact_1 rtol = 0.25

CMI_hist = InformationTheory.c_mutual(InformationTheory.ConditionalMutualInformation.Hist(), x_data, y_data₂, z_data)
println("Exact CMI(X;Y|Z): $CMI_exact_2\t Histogram CMI(X;Y|Z): $CMI_hist\t error: $(abs(CMI_hist - CMI_exact_2))")
@test CMI_hist ≈ CMI_exact_2 rtol = 0.25

println("---------- kNN estimator method ----------")

# Estimate entropy using a kNN estimator
CMI_knn = InformationTheory.c_mutual(InformationTheory.ConditionalMutualInformation.kNN(), x_data, y_data₁, z_data)
println("Exact CMI(X;Y|Z): $CMI_exact_1\t KNN CMI(X;Y|Z): $CMI_knn\t error: $(abs(CMI_knn - CMI_exact_1))")
@test CMI_knn ≈ CMI_exact_1 rtol = 0.5

CMI_knn = InformationTheory.c_mutual(InformationTheory.ConditionalMutualInformation.kNN(), x_data, y_data₂, z_data)
println("Exact CMI(X;Y|Z): $CMI_exact_2\t KNN CMI(X;Y|Z): $CMI_knn\t error: $(abs(CMI_knn - CMI_exact_2))")
@test CMI_knn ≈ CMI_exact_2 rtol = 0.5

# Reference
H_xz = InformationTheory.shannon(InformationTheory.ShannonEntropy.KSG(), x_data, z_data)
H_yz = InformationTheory.shannon(InformationTheory.ShannonEntropy.KSG(), y_data₁, z_data)
H_z = InformationTheory.shannon(InformationTheory.ShannonEntropy.KSG(), z_data)
H_xyz = InformationTheory.shannon(InformationTheory.ShannonEntropy.KSG(), x_data, y_data₁, z_data)
CMI_ref = H_xz + H_yz - H_z - H_xyz
println("Exact CMI(X;Y|Z): $CMI_exact_1\t Ref CMI(X;Y|Z): $CMI_ref\t error: $(abs(CMI_ref - CMI_exact_1))")
@test CMI_ref ≈ CMI_exact_1 rtol = 0.1
