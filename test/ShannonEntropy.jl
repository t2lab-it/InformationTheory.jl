# Generate random data
x_data = randn(10000)
y_data = randn(10000)

# Exact entropy for a standard normal distribution
hx_exact = 0.5 * (1.0 + log(2π))
hxy_exact = 2.0 * hx_exact

println("---------- histogram method ----------")

# Estimate entropy using a histogram
hx_hist = shannon(ShannonEntropy.Hist(), x_data)
println("Exact H(X): $hx_exact\t\t Histogram H(X): $hx_hist\t error: $(abs(hx_hist - hx_exact))")
@test hx_hist ≈ hx_exact rtol = 0.1

# Estimate joint entropy using a histogram
hxy_hist = shannon(ShannonEntropy.Hist(), x_data, y_data)
println("Exact H(X,Y): $hxy_exact\t Histogram H(X,Y): $hxy_hist\t error: $(abs(hxy_hist - hxy_exact))")
@test hxy_hist ≈ hxy_exact rtol = 0.1

# Estimate joint entropy with custom histogram bins
hxy_hist_bins = shannon(ShannonEntropy.Hist(bins=([-4:0.25:4;],-5:0.25:5)), x_data, y_data)
println("Exact H(X,Y): $hxy_exact\t Histogram H(X,Y): $hxy_hist_bins\t error: $(abs(hxy_hist_bins - hxy_exact))")
@test hxy_hist_bins ≈ hxy_exact rtol = 0.1

println("---------- ksg estimator method ----------")

# Estimate entropy using a ksg estimator
hx_ksg = shannon(ShannonEntropy.KSG(), x_data)
println("Exact H(X): $hx_exact\t\t KSG H(X): $hx_ksg\t\t error: $(abs(hx_ksg - hx_exact))")
@test hx_ksg ≈ hx_exact rtol = 0.1

# Estimate joint entropy using a ksg estimator
hxy_ksg = shannon(ShannonEntropy.KSG(), x_data, y_data)
println("Exact H(X,Y): $hxy_exact\t KSG H(X,Y): $hxy_ksg\t\t error: $(abs(hxy_ksg - hxy_exact))")
@test hxy_ksg ≈ hxy_exact rtol = 0.1

# Estimate joint entropy with custom k value
hxy_ksg_custom = shannon(ShannonEntropy.KSG(k=10), x_data, y_data)
println("Exact H(X,Y): $hxy_exact\t KSG H(X,Y): $hxy_ksg_custom\t\t error: $(abs(hxy_ksg_custom - hxy_exact))")
@test hxy_ksg_custom ≈ hxy_exact rtol = 0.1
