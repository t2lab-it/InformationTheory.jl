# KL divergence

このページでは，カルバック・ライブラー情報量 (Kullback-Leibler divergence) の計算関数`kldiv`について説明します．

## カルバック・ライブラー情報量

2つの確率分布$P$と$Q$に対するカルバック・ライブラー情報量
```math
    D_{KL}(P||Q) = \int_{-\infty}^{\infty} p(x) \ln \frac{p(x)}{q(x)} \ \mathrm{d}x
```
です．

### ヒストグラムを使った計算例

以下は，ヒストグラム推定を使った計算例です．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000) # P
y = randn(10000) # Q

# ヒストグラム推定を使うことを指定
est = KLdivergence.Hist()

# KLダイバージェンスの推定
kl = kldiv(est, x, y)

# 結果の出力
println(kl)
```

ヒストグラムを使用する際には，`bins_x`と`bins_y`を指定できます．

```julia
est = KLdivergence.Hist(bins_x = (-5:0.25:5,), bins_y = (-5:0.25:5,))
```

### k-近傍法を使った計算例

k-近傍法は，Wang et. al.(2009)が提案した手法に基づいて計算されます．
論文中ではいくつかの手法を提案していますが，ここでは
```math
    D_{KL}(P||Q) = \frac{1}{N} \sum_{i=1}^N \left[\psi(\ell_i) - \psi(k_i)\right] + \ln \frac{M-1}{N}
```
で計算します．
$d$は次元，$N$は$P$のデータ数，$M$は$Q$のデータ数です．
$\ell_i$は$P$の$i$番目の点から$P$の$k$番目に近い点までの距離，$k_i$は$P$の$i$番目の点から$Q$の$k$番目に近い点までの距離を表します．
この距離は，`NearestNeighbors.jl`のkd木を使用して高速に推定されます．
距離推定には論文で提示されている通り，Chebyshev距離を用いています．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000) # P
y = randn(10000) # Q

# KSG推定を使うことを指定
est = KLdivergence.kNN()

# KLダイバージェンスの推定
kl = kldiv(est, x, y)

# 結果の出力
println(kl)
```

k-近傍法では，`k`を指定できます．
```julia
est = KLdivergence.kNN(k = 10)
```

## 高次元のKL情報量

同じ関数を利用して，より高次元の確率分布に対するKLダイバージェンス
```math
    D_{KL}(P(X_1,Y_1,Z_1,\dots)||Q(X_2,Y_2,Z_2,\dots))
```
も推定可能です．

### ヒストグラムを使った計算例

ここでは，$D_{KL}(P(X,Y)||Q(X,Y))$を推定する例を示します．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x1 = randn(10000)
x2 = randn(10000)
y1 = randn(10000)
y2 = randn(10000)

# ヒストグラム推定を使うことを指定
est = KLdivergence.Hist()

# KLダイバージェンスの推定
kl = kldiv(est, (x1, x2), (y1, y2))

# 結果の出力
println(kl)
```

### k-近傍法を使った計算例

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x1 = randn(10000)
x2 = randn(10000)
y1 = randn(10000)
y2 = randn(10000)

# KSG推定を使うことを指定
est = KLdivergence.kNN()

# KLダイバージェンスの推定
kl = kldiv(est, (x1, x2), (y1, y2))

# 結果の出力
println(kl)
```

次元の数には制限ありませんが，高次元になるほど推定精度は下がり，推定時間も伸びます．

## 参考文献

- Wang Q. , Kulkarni S. R., Verdu S., Divergence Estimation for Multidimensional Densities Via  k -Nearest-Neighbor Distances, in IEEE Transactions on Information Theory, Vol. 55, No. 5 (2009), pp. 2392-2405
