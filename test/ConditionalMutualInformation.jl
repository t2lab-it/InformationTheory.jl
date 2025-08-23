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
MI_hist = InformationTheory.c_mutual(InformationTheory.ConditionalMutualInformation.Hist(), x_data, y_data₁, z_data)
println("Exact CMI(X;Y|Z): $CMI_exact_1\t Histogram CMI(X;Y|Z): $MI_hist\t error: $(abs(MI_hist - CMI_exact_1))")
@test MI_hist ≈ CMI_exact_1 rtol = 0.25

MI_hist = InformationTheory.c_mutual(InformationTheory.ConditionalMutualInformation.Hist(), x_data, y_data₂, z_data)
println("Exact CMI(X;Y|Z): $CMI_exact_2\t Histogram CMI(X;Y|Z): $MI_hist\t error: $(abs(MI_hist - CMI_exact_2))")
@test MI_hist ≈ CMI_exact_2 rtol = 0.1
