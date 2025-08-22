# Generate random data
x_data = randn(10000)
y_data = randn(10000)

# Exact entropy for a standard normal distribution
hx_exact = 0.5 * (1.0 + log(2π))
hxy_exact = 2.0 * hx_exact

# Estimate entropy using a histogram
hx_hist = InformationTheory.shannon(InformationTheory.Hist(), x_data)
println("Exact H(X): $hx_exact\t\t Histogram H(X): $hx_hist")
@test hx_hist ≈ hx_exact rtol = 0.1

# Estimate joint entropy using a histogram
hxy_hist = InformationTheory.shannon(InformationTheory.Hist(), x_data, y_data)
println("Exact H(X,Y): $hxy_exact\t Histogram H(X,Y): $hxy_hist")
@test hxy_hist ≈ hxy_exact rtol = 0.1

# Estimate joint entropy with custom histogram bins
hxy_hist_bins = InformationTheory.shannon(InformationTheory.Hist(bins=([-4:0.25:4;],-5:0.25:5)), x_data, y_data)
println("Exact H(X,Y): $hxy_exact\t Histogram H(X,Y): $hxy_hist_bins")
@test hxy_hist_bins ≈ hxy_exact rtol = 0.1