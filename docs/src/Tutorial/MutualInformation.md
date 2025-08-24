# Mutual Information

このページでは，相互情報量 (mutual information) の計算関数`mutual`について説明します．

## 相互情報量

2変数に対する相互情報量
```math
    I(X:Y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} p(x, y) \ln \frac{p(x, y)}{p(x)p(y)} \ \mathrm{d}x \mathrm{d}y
```
です．

### ヒストグラムを使った計算例

以下は，ヒストグラム推定を使った計算例です．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000)
y = randn(10000)

# ヒストグラム推定を使うことを指定
est = MutualInformation.Hist()

# 相互情報量の推定
mi = mutual(est, x, y)

# 結果の出力
println(mi)
```

ヒストグラムを使用する際には，`bins`を指定できます．

```julia
est = MutualInformation.Hist(bins = (-5:0.25:5, -5:0.25:5))
```

### k-近傍法を使った計算例

k-近傍法は，Kraskov et. al.(2004)が提案した手法に基づいて計算されます．
論文中では$I^{(1)}$と$I^{(2)}$が提案されていますが，ここでは$I^{(1)}$の方法を採用して，
```math
    I(X:Y) = \psi(k) + \psi(N)- \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle
```
で計算します．
$\psi$はディガンマ関数で，$N$はデータ数です．
$n_x$と$n_y$は，各点の最近傍距離$\epsilon_k$内に存在する点の数を表します．
この距離は，`NearestNeighbors.jl`のkd木を使用して高速に推定されます．
距離推定には論文で提示されている通り，Chebyshev距離を用いています．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000)
y = randn(10000)

# KSG推定を使うことを指定
est = MutualInformation.KSG()

# 相互情報量の推定
mi = mutual(est, x, y)

# 結果の出力
println(mi)
```

k-近傍法では，`k`を指定できます．
```julia
est = MutualInformation.KSG(k = 10)
```

## 高次元の相互情報量

より高次元の相互情報量
```math
    I([X_1, X_2, \dots] : [Y_1, Y_2, \dots])
```
についても推定可能です．

### ヒストグラムを使った計算例

ここでは，$I([X_1, X_2]:[Y_1,Y_2])$を推定する例を示します．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x1 = randn(10000)
x2 = randn(10000)
y1 = randn(10000)
y2 = randn(10000)

# ヒストグラム推定を使うことを指定
est = MutualInformation.Hist()

# 相互情報量の推定
mi = mutual(est, (x1, x2), (y1, y2))

# 結果の出力
println(mi)
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
est = MutualInformation.KSG()

# 相互情報量の推定
mi = mutual(est, (x1, x2), (y1, y2))

# 結果の出力
println(mi)
```

次元の数には制限ありませんが，高次元になるほど推定精度は下がり，推定時間も伸びます．

## 参考文献

- Kraskov A., Stögbauer H., Grassberger P., Estimating mutual information, Phys. Rev. E, Vol. 69 (2004), p. 066138.
