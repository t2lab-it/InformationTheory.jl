# Generate random data
x_data = randn(50000)
y_data = randn(50000) .+ 1.0
z_data = randn(50000) * 2.0

# Exact entropy for a standard normal distribution
kl_exact = 0.5
kl_exact2 = -3.0/8.0 + log(2.0)

println("---------- histogram method ----------")

# Estimate entropy using a histogram
kl_hist = kldiv(KLdivergence.Hist(), x_data, y_data)
println("Exact D(p||q): $kl_exact\t\t\t Histogram D(p||q): $kl_hist\t error: $(abs(kl_hist - kl_exact))")
@test kl_hist ≈ kl_exact rtol = 0.1

kl_hist = kldiv(KLdivergence.Hist(), x_data, z_data)
println("Exact D(p||q): $kl_exact2\t Histogram D(p||q): $kl_hist\t error: $(abs(kl_hist - kl_exact2))")
@test kl_hist ≈ kl_exact2 rtol = 0.1

# Estimate entropy with custom histogram bins
kl_hist_bins = kldiv(KLdivergence.Hist(bins_x=(-5:0.25:5,), bins_y=(-5:0.25:5,)), x_data, y_data)
println("Exact D(p||q): $kl_exact\t\t\t Histogram D(p||q): $kl_hist_bins\t error: $(abs(kl_hist_bins - kl_exact))")
@test kl_hist_bins ≈ kl_exact rtol = 0.1

println("---------- kNN estimator method ----------")

# Estimate entropy using a kNN estimator
kl_kNN = kldiv(KLdivergence.kNN(k = 10), x_data, y_data)
println("Exact D(p||q): $kl_exact\t\t\t kNN D(p||q): $kl_kNN\t error: $(abs(kl_kNN - kl_exact))")
@test kl_kNN ≈ kl_exact rtol = 0.1

kl_kNN = kldiv(KLdivergence.kNN(k = 10), x_data, z_data)
println("Exact D(p||q): $kl_exact2\t kNN D(p||q): $kl_kNN\t error: $(abs(kl_kNN - kl_exact2))")
@test kl_kNN ≈ kl_exact2 rtol = 0.25
