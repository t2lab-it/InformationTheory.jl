# Generate random data
x_data = randn(10000)
y_data = randn(10000) .+ 1.0

# Exact entropy for a standard normal distribution
kl_exact = 0.5

println("---------- histogram method ----------")

# Estimate entropy using a histogram
kl_hist = InformationTheory.kldiv(InformationTheory.KLdivergence.Hist(), x_data, y_data)
println("Exact D(p||q): $kl_exact\t\t Histogram D(p||q): $kl_hist\t error: $(abs(kl_hist - kl_exact))")
@test kl_hist ≈ kl_exact rtol = 0.1

# Estimate entropy with custom histogram bins
kl_hist_bins = InformationTheory.kldiv(InformationTheory.KLdivergence.Hist(bins_x=(-5:0.25:5,), bins_y=(-5:0.25:5,)), x_data, y_data)
println("Exact D(p||q): $kl_exact\t Histogram D(p||q): $kl_hist_bins\t error: $(abs(kl_hist_bins - kl_exact))")
@test kl_hist_bins ≈ kl_exact rtol = 0.1
