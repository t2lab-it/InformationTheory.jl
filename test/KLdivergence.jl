# Generate random data
x_data = randn(50000)
y_data = randn(50000) .+ 1.0
z_data = randn(50000) * 2.0

# Exact entropy for a standard normal distribution
kl_exact = 0.5
kl_exact2 = -3.0/8.0 + log(2.0)

println("---------- histogram method ----------")

# Estimate entropy using a histogram
kl_hist = InformationTheory.kldiv(InformationTheory.KLdivergence.Hist(), x_data, y_data)
println("Exact D(p||q): $kl_exact\t\t\t Histogram D(p||q): $kl_hist\t error: $(abs(kl_hist - kl_exact))")
@test kl_hist ≈ kl_exact rtol = 0.1

kl_hist = InformationTheory.kldiv(InformationTheory.KLdivergence.Hist(), x_data, z_data)
println("Exact D(p||q): $kl_exact2\t Histogram D(p||q): $kl_hist\t error: $(abs(kl_hist - kl_exact2))")
@test kl_hist ≈ kl_exact2 rtol = 0.1

# Estimate entropy with custom histogram bins
kl_hist_bins = InformationTheory.kldiv(InformationTheory.KLdivergence.Hist(bins_x=(-5:0.25:5,), bins_y=(-5:0.25:5,)), x_data, y_data)
println("Exact D(p||q): $kl_exact\t\t\t Histogram D(p||q): $kl_hist_bins\t error: $(abs(kl_hist_bins - kl_exact))")
@test kl_hist_bins ≈ kl_exact rtol = 0.1

println("---------- ksg estimator method ----------")

# Estimate entropy using a ksg estimator
kl_ksg = InformationTheory.kldiv(InformationTheory.KLdivergence.KSG(k = 10), x_data, y_data)
println("Exact D(p||q): $kl_exact\t\t\t KSG D(p||q): $kl_ksg\t error: $(abs(kl_ksg - kl_exact))")
@test kl_ksg ≈ kl_exact rtol = 0.1

kl_ksg = InformationTheory.kldiv(InformationTheory.KLdivergence.KSG(k = 10), x_data, z_data)
println("Exact D(p||q): $kl_exact2\t KSG D(p||q): $kl_ksg\t error: $(abs(kl_ksg - kl_exact2))")
@test kl_ksg ≈ kl_exact2 rtol = 0.1
