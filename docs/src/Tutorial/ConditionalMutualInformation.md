# Conditional Mutual Information

このページでは，条件付き相互情報量 (conditional mutual information) の計算関数`c_mutual`について説明します．

## 条件付き相互情報量

3つの確率変数$X, Y, Z$に対する条件付き相互情報量
```math
    I(X : Y | Z) = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\int_{-\infty}^{\infty} p(x, y, z) \log \frac{p(x, y | z)}{p(x | z) p(y | z)}
```
です．

### ヒストグラムを使った計算例

以下は，ヒストグラム推定を使った計算例です．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000)
y = randn(10000)
z = randn(10000)

# ヒストグラム推定を使うことを指定
est = ConditionalMutualInformation.Hist()

# 条件付き相互情報量の推定
cmi = c_mutual(est, x, y, z)

# 結果の出力
println(cmi)
```

ヒストグラムを使用する際には，`bins_x`, `bins_y`, `bins_z`を指定できます．

```julia
est = ConditionalMutualInformation.Hist(bins_x = (-5:0.25:5,), bins_y = (-5:0.25:5,), bins_z = (-5:0.25:5,))
```

### k-近傍法を使った計算例

k-近傍法は，Kraskov et. al.(2004)が提案した手法に基づいて計算されます．
ここでは，
```math
    I(X : Y | Z) = \psi(k) + \langle \psi(n_z + 1) - \psi(n_{x,z} + 1) - \psi(n_{y,z} + 1) \rangle
```
で計算します．
$\psi$はディガンマ関数で，$k$は最近傍の数です．
$n_z, n_{x,z}, n_{y,z}$は、それぞれ$Z, (X, Z), (Y, Z)$空間における最近傍の数を表します。
この距離は，`NearestNeighbors.jl`のkd木を使用して高速に推定されます．
距離推定には論文で提示されている通り，Chebyshev距離を用いています．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000)
y = randn(10000)
z = randn(10000)

# KSG推定を使うことを指定
est = ConditionalMutualInformation.kNN()

# 条件付き相互情報量の推定
cmi = c_mutual(est, x, y, z)

# 結果の出力
println(cmi)
```

k-近傍法では，`k`を指定できます．
```julia
est = ConditionalMutualInformation.kNN(k = 10)
```

## 高次元の確率分布

同じ関数を利用して，より高次元の確率分布に対する条件付き相互情報量も推定可能です．
ここでは，$I([X_1, X_2] : [Y_1, Y_2] | [Z_1, Z_2])$を推定する例を示します．

### ヒストグラムを使った計算例

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x1 = randn(10000)
x2 = randn(10000)
y1 = randn(10000)
y2 = randn(10000)
z1 = randn(10000)
z2 = randn(10000)

# ヒストグラム推定を使うことを指定
est = ConditionalMutualInformation.Hist()

# 条件付き相互情報量の推定
cmi = c_mutual(est, (x1, x2), (y1, y2), (z1, z2))

# 結果の出力
println(cmi)
```

### k-近傍法を使った計算例

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x1 = randn(10000)
x2 = randn(10000)
y1 = randn(10000)
y2 = randn(10000)
z1 = randn(10000)
z2 = randn(10000)

# KSG推定を使うことを指定
est = ConditionalMutualInformation.kNN()

# 条件付き相互情報量の推定
cmi = c_mutual(est, (x1, x2), (y1, y2), (z1, z2))

# 結果の出力
println(cmi)
```

次元の数には制限ありませんが，高次元になるほど推定精度は下がり，推定時間も伸びます．

## 参考文献

- Kraskov A., Stögbauer H., Grassberger P., Estimating mutual information, Phys. Rev. E, Vol. 69 (2004), p. 066138.
