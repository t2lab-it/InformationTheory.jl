
# Shannon Entropy

このページでは，シャノンエントロピーの計算関数`shannon`について説明します．
この関数は，通常のシャノンエントロピーだけでなく，結合エントロピーも計算可能です．

## 平均情報量

最も簡単な例は，1変数に対するシャノンエントロピー（平均情報量）
```math
    H(X) = \int_{-\infty}^{\infty} - p(x) \ln p(x) \ \mathrm{d}x
```
です．


### ヒストグラムを使った計算例

以下は，ヒストグラム推定を使った計算例です．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000)

# ヒストグラム推定を使うことを指定
est = ShannonEntropy.Hist()

# シャノンエントロピーの推定
hx = shannon(est, x)

# 結果の出力
println(hx)
```

ヒストグラムを使用する際には，`bins`を指定できます．

```julia
est = ShannonEntropy.Hist(bins = (-5:0.25:5,))
```

### k-近傍法を使った計算例

k-近傍法は，Kraskov et. al.(2004)が提案した手法に基づいて計算されます．
ここでは，
```math
    H(X) = -\psi(k) + \psi(N) + \frac{d}{N}\sum_{i=1}^N \ln\epsilon_i
```
で計算します．
$\psi$はディガンマ関数で，$d$は確率変数$X$の次元，$N$はデータ数です．
$\epsilon_i$は各点から$k$番目に近い点までの距離を表しています．
この距離は，`NearestNeighbors.jl`のkd木を使用して高速に推定されます．
距離推定には論文で提示されている通り，Chebyshev距離を用いています．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000)

# KSG推定を使うことを指定
est = ShannonEntropy.KSG()

# シャノンエントロピーの推定
hx = shannon(est, x)

# 結果の出力
println(hx)
```

k-近傍法では，`k`を指定できます．
```julia
est = ShannonEntropy.KSG(k = 10)
```

## 結合エントロピー

同じ関数を利用して，より高次元の結合エントロピー
```math
    H(X, Y, Z, ...)
```
についても推定可能です．

### ヒストグラムを使った計算例

ここでは，$H(X,Y)$を推定する例を示します．

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000)
y = randn(10000)

# ヒストグラム推定を使うことを指定
est = ShannonEntropy.Hist()

# シャノンエントロピーの推定
hx = shannon(est, x, y)

# 結果の出力
println(hx)
```

### k-近傍法を使った計算例

```julia
using InformationTheory

# 正規分布に基づくランダムデータを生成
x = randn(10000)
y = randn(10000)

# KSG推定を使うことを指定
est = ShannonEntropy.KSG()

# シャノンエントロピーの推定
hx = shannon(est, x, y)

# 結果の出力
println(hx)
```

引数の数（結合数）には制限ありませんが，高次元になるほど推定精度は下がり，推定時間も伸びます．

## 参考文献

- Kraskov A., Stögbauer H., Grassberger P., Estimating mutual information, Phys. Rev. E, Vol. 69 (2004), p. 066138.
