# 线性模型

## 基本形式
给定由 $d$ 个属性描述的示例 $\boldsymbol{x} = (x_1; x_2; \dots; x_d)$，其中 $x_i$ 是 $\boldsymbol{x}$ 在第 $i$ 个属性上的取值，线性模型（linear model）试图学得一个通过属性的线性组合来进行预测的函数，即

$$f(\boldsymbol{x}) = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b , \tag{3.1}$$

一般用向量形式写成

$$f(\boldsymbol{x}) = \boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b , \tag{3.2}$$

其中 $\boldsymbol{w} = (w_1; w_2; \dots; w_d)$。$\boldsymbol{w}$ 和 $b$ 学得之后，模型就得以确定

## 线性回归

### 基本定义
给定数据集
$$D=\{(\boldsymbol x_1,y_1),(\boldsymbol x_2,y_2)\dots(\boldsymbol x_m,y_m)\}$$
线性回归试图学一个线性模型尽可能准确预测输出标记
$$f(x_i)=wx_i+b 使得f(x_i)尽可能等于y_i$$

对于离散属性：
* 若存在序的关系：可通过连续化转换为连续值
* 不存在序的关系：则可以转化为k维向量

### 损失函数

回归任务中最常用的损失函数是均方误差MSE
$$(w^*, b^*) = \mathop{\arg\min}_{(w,b)} \sum_{i=1}^{m} (f(x_i) - y_i)^2 = \mathop{\arg\min}_{(w,b)} \sum_{i=1}^{m} (y_i - w x_i - b)^2 .$$
基于均方误差最小化求解模型：**最小二乘法**

尝试找到一条直线，使样本点到直线欧氏距离最小
### 假设只有一个属性
求解 $w$ 和 $b$ 使 $E_{(w,b)} = \sum_{i=1}^{m} (y_i - w x_i - b)^2$ 最小

我们可将 $E_{(w,b)}$ 分别对 $w$ 和 $b$ 求导得到

$$\frac{\partial E_{(w,b)}}{\partial w} = 2 \left( w \sum_{i=1}^{m} x_i^2 - \sum_{i=1}^{m} (y_i - b) x_i \right) , \tag{3.5}$$

$$\frac{\partial E_{(w,b)}}{\partial b} = 2 \left( m b - \sum_{i=1}^{m} (y_i - w x_i) \right) , \tag{3.6}$$

然后令式 (3.5) 和 (3.6) 为零可得到 $w$ 和 $b$ 最优解的闭式解

$$w = \frac{\sum_{i=1}^{m} y_i (x_i - \bar{x})}{\sum_{i=1}^{m} x_i^2 - \frac{1}{m} \left( \sum_{i=1}^{m} x_i \right)^2} , \tag{3.7}$$

$$b = \frac{1}{m} \sum_{i=1}^{m} (y_i - w x_i) , \tag{3.8}$$

其中 $\bar{x} = \frac{1}{m} \sum_{i=1}^{m} x_i$ 为 $x$ 的均值

**为什么是全局最小值：因为这个损失函数是严格凸函数**
### 正常情况有d个属性：多元线性回归
类似的可利用最小二乘法来对 $\boldsymbol{w}$ 和 $b$ 进行估计。为便于讨论，我们把 $\boldsymbol{w}$ 和 $b$ 吸收入向量形式 $\hat{\boldsymbol{w}} = (\boldsymbol{w}; b)$，相应的，把数据集 $D$ 表示为一个 $m \times (d + 1)$ 大小的矩阵 $\mathbf{X}$

$$\mathbf{X} = \begin{pmatrix}
x_{11} & x_{12} & \dots & x_{1d} & 1 \\
x_{21} & x_{22} & \dots & x_{2d} & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
x_{m1} & x_{m2} & \dots & x_{md} & 1
\end{pmatrix} = \begin{pmatrix}
\boldsymbol{x}_1^{\mathrm{T}} & 1 \\
\boldsymbol{x}_2^{\mathrm{T}} & 1 \\
\vdots & \vdots \\
\boldsymbol{x}_m^{\mathrm{T}} & 1
\end{pmatrix} ,$$

再把标记也写成向量形式 $\boldsymbol{y} = (y_1; y_2; \dots; y_m)$

$$\hat{\boldsymbol{w}}^* = \mathop{\arg\min}_{\hat{\boldsymbol{w}}} (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})^{\mathrm{T}}(\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}}) . \tag{3.9}$$

令 $E_{\hat{\boldsymbol{w}}} = (\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})^{\mathrm{T}}(\boldsymbol{y} - \mathbf{X}\hat{\boldsymbol{w}})$，对 $\hat{\boldsymbol{w}}$ 求导得到

$$\frac{\partial E_{\hat{\boldsymbol{w}}}}{\partial \hat{\boldsymbol{w}}} = 2\mathbf{X}^{\mathrm{T}}(\mathbf{X}\hat{\boldsymbol{w}} - \boldsymbol{y}) . \tag{3.10}$$

令上式为零可得 $\hat{\boldsymbol{w}}$ 最优解的闭式解，但由于涉及矩阵逆的计算，下面做一个简单的讨论
* 当 $\mathbf{X}^{\mathrm{T}}\mathbf{X}$ 为满秩矩阵或正定矩阵时，令式 (3.10) 为零可得 
    $$\hat{\boldsymbol{w}}^* = (\mathbf{X}^{\mathrm{T}}\mathbf{X})^{-1}\mathbf{X}^{\mathrm{T}}\boldsymbol{y} , \tag{3.11}$$
    令 $\hat{\boldsymbol{x}}_i = (\boldsymbol{x}_i, 1)$，则最终学得的多元线性回归模型为
    $$f(\hat{\boldsymbol{x}}_i) = \hat{\boldsymbol{x}}_i^{\mathrm{T}}(\mathbf{X}^{\mathrm{T}}\mathbf{X})^{-1}\mathbf{X}^{\mathrm{T}}\boldsymbol{y} . \tag{3.
    12}$$
* 其他情况：会解得多个***W***
### 对数线性回归
假定输入特征向量对应的输出向量在指数尺度上变化：本质是输入空间到输出空间的非线性映射
$$\ln y = \boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b . \tag{3.14}$$
更一般地，考虑单调可微函数 $g(\cdot)$，令
$$y = g^{-1}(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b) , \tag{3.15}$$
这样得到的模型称为“广义线性模型”（generalized linear model），其中函数 $g(\cdot)$ 称为“联系函数”（link function）

## 对数几率回归
如何使用线性模型完成分类任务：使用广义回归模型（单调可微函数）将真实类别和模型预测相联系

### 二分类问题

#### 1.确定映射函数
类别标记$$y\in\{0,1\}$$
输出实值$$z = \boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b , $$
我们希望得到一个介于`[0,1]`之间的正例概率：引入logistic函数（阶跃函数性质不好）

**这里注意：并不是让模型直接输出`{0,1}`的离散标签，而是输出一个概率结合人为设定的阈值(一般是0.5)判断正例反例**
$$y = \frac{1}{1 + e^{-z}} \ . \tag{3.17}$$
$$y = \frac{1}{1 + e^{-(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b)}} \ . \tag{3.18}$$
类似于式 (3.14)，式 (3.18) 可变化为
$$\ln \frac{y}{1 - y} = \boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b \ . \tag{3.19}$$
若将 $y$ 视为样本 $\boldsymbol{x}$ 作为正例的可能性，则 $1 - y$ 是其反例可能性
$$\frac{y}{1 - y} \tag{3.20}$$
两者的比值称为“几率”（odds），反映了 $\boldsymbol{x}$ 作为正例的相对可能性
**这一段的核心逻辑：经过Sigmoid变换后的概率，其对数几率是一个线性函数**

---
**换个角度出发：我们考虑能否直接让正例概率和特征之间建立线性关系**
$$P(y=1\mid \mathbf x)=\mathbf w^T\mathbf x+b$$
很明显左侧值域`[0,1]`但是右侧是实数域，需要对左侧概率进行某种变形
$$Odds=\frac{p}{1-p}$$
但是几率没有办法表达负数，我们再对几率取对数
$$\boxed{\ln\frac{p}{1-p}=\mathbf w^T\mathbf x+b}$$
将这个公式化见化简就能得到Sigmoid函数

---
优点：
1. 直接对分类可能做建模，无需提前知道数据分布
3. 任意阶可导凸函数，具有优秀的数学性质
2. 可以得到近似概率预测
#### 2.利用极大似然估计得到表达式
将对数几率带入MSE得到的损失函数是非凸函数，对于逻辑回归函数，采用负对数似然（交叉熵）是更好的选择

**我们约定p:样本为正例的概率**

1. **单个样本的似然项：**
对于第 $i$ 个样本 $(\boldsymbol{x}_i, y_i)$，标签 $y_i \in \{0, 1\}$：
$$p_1=P(y=1\mid x)=p$$ $$p_0=P(y=0\mid x)=1-p$$
可以把两种情况合并成一个公式：
$$\boxed{P(y\mid x)=y p+(1-y)(1-p)}(1)$$ $$\boxed{P(y\mid x)=p^y(1-p)^{1-y}}(2)$$
根据Sigmoid函数我们已知正例概率：
$$p=\frac{1}{1+e^{-z}}=\frac{e^z}{1+e^z}$$
将正例概率带入(2)表达式可以得到单个样本对数似然：
$$\boxed{\ln P(y\mid x)=yz-\ln(1+e^z)}$$

2. **所有样本的联合概率（似然函数）：**
所有样本同时出现的似然，就是各样本概率相乘：
$$L(\mathbf w,b)=\prod_{i=1}^{m}P(y_i\mid x_i)$$


3. **对数似然：**
取自然对数，乘法就变成加法：
$$\ell(\mathbf w,b)=\sum_{i=1}^{m}\ln P(y_i\mid x_i)$$
**极大似然估计的目标就是调节 $\boldsymbol{w}$ 和 $b$，让这个总对数概率 $\ell(\boldsymbol{w}, b)$ 达到最大**

#### 3.求解最终参数
将最大化似然取负号等价于最小化损失
$$J(\mathbf w,b)=-\sum_{i=1}^{m}\ln P(y_i\mid x_i)$$
$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^{m} \left( -y_i \boldsymbol{\beta}^{\mathrm{T}}\hat{\boldsymbol{x}}_i + \ln(1 + e^{\boldsymbol{\beta}^{\mathrm{T}}\hat{\boldsymbol{x}}_i}) \right)$$
[数学优化：最小化损失函数](chapter3_note.md)

## 线性判别分析
### 二分类
核心思想：给定训练集，设法将样例投影到一条直线上,使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离

在对新样本进行分类时，将其投影到同样的这条直线上，再根据投影点的位置来确定新样本的类别

定义数学符号
1. 数据集 $D = \{(\boldsymbol{x}_i, y_i)\}_{i=1}^m$ $y_i \in \{0, 1\}$ 
2. 第i类集合 $X_i$ $i \in \{0, 1\}$
2. 第i类的均值向量 $\boldsymbol{\mu}_i$ $i \in \{0, 1\}$
4. 第i类的散度矩阵 $\boldsymbol{\Sigma}_i$ $i \in \{0, 1\}$

若将数据投影到直线 $\boldsymbol{w}$ 上，则两类样本的中心在直线上的投影分别为 $\boldsymbol{w}^{\mathrm{T}}\boldsymbol{\mu}_0$ 和 $\boldsymbol{w}^{\mathrm{T}}\boldsymbol{\mu}_1$

若将所有样本点都投影到直线上，则两类样本的协方差分别为 $\boldsymbol{w}^{\mathrm{T}}\boldsymbol{\Sigma}_0\boldsymbol{w}$ 和 $\boldsymbol{w}^{\mathrm{T}}\boldsymbol{\Sigma}_1\boldsymbol{w}$

欲使同类样例的投影点尽可能接近，可以让同类样例投影点的协方差尽可能小，即 $\boldsymbol{w}^{\mathrm{T}}\boldsymbol{\Sigma}_0\boldsymbol{w} + \boldsymbol{w}^{\mathrm{T}}\boldsymbol{\Sigma}_1\boldsymbol{w}$ 尽可能小；而欲使异类样例的投影点尽可能远离，可以让类中心之间的距离尽可能大，即 $\|\boldsymbol{w}^{\mathrm{T}}\boldsymbol{\mu}_0 - \boldsymbol{w}^{\mathrm{T}}\boldsymbol{\mu}_1\|_2^2$ 尽可能大，则可得到欲最大化的目标
$$\begin{aligned}J &= \frac{\|\boldsymbol{w}^{\mathrm{T}}\boldsymbol{\mu}_0 - \boldsymbol{w}^{\mathrm{T}}\boldsymbol{\mu}_1\|_2^2}{\boldsymbol{w}^{\mathrm{T}}\boldsymbol{\Sigma}_0\boldsymbol{w} + \boldsymbol{w}^{\mathrm{T}}\boldsymbol{\Sigma}_1\boldsymbol{w}} \\
&= \frac{\boldsymbol{w}^{\mathrm{T}}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^{\mathrm{T}}\boldsymbol{w}}{\boldsymbol{w}^{\mathrm{T}}(\boldsymbol{\Sigma}_0 +\boldsymbol{\Sigma}_1)\boldsymbol{w}} \end{aligned}$$
由此引出类内散度矩阵和类间散度矩阵
$$\begin{aligned}
\mathbf{S}_w &= \boldsymbol{\Sigma}_0 + \boldsymbol{\Sigma}_1 \\
&= \sum_{\boldsymbol{x} \in X_0} (\boldsymbol{x} - \boldsymbol{\mu}_0)(\boldsymbol{x} - \boldsymbol{\mu}_0)^{\mathrm{T}} + \sum_{\boldsymbol{x} \in X_1} (\boldsymbol{x} - \boldsymbol{\mu}_1)(\boldsymbol{x} - \boldsymbol{\mu}_1)^{\mathrm{T}} \tag{3.33}\end{aligned}$$
$$\mathbf{S}_b = (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^{\mathrm{T}} \,, \tag{3.34}$$
到这里优化函数最终变为：
$$J = \frac{\boldsymbol{w}^{\mathrm{T}}\mathbf{S}_b\boldsymbol{w}}{\boldsymbol{w}^{\mathrm{T}}\mathbf{S}_w\boldsymbol{w}}$$
由于优化函数是上下关于 $w$ 的二次式，任意倍乘 $aw$ 都可以约分，这说明 $J$ 只和 $w$ 的方向有关和长度无关，人为将分母设定1转换为带约束的最优化问题
$$\begin{aligned}\min_{\boldsymbol{w}} \quad & -\boldsymbol{w}^{\mathrm{T}}\mathbf{S}_b\boldsymbol{w} \\
\text{s.t.} \quad & \boldsymbol{w}^{\mathrm{T}}\mathbf{S}_w\boldsymbol{w} = 1 \end{aligned}$$
构造拉格朗日函数后对 $w$ 求偏导并且让偏导结果为0最后化简得到
$$L(\boldsymbol{w}, \lambda) = -\boldsymbol{w}^T \mathbf{S}_b \boldsymbol{w} + \lambda (\boldsymbol{w}^T \mathbf{S}_w \boldsymbol{w} - 1)$$
$$\boxed{\mathbf{S}_b \boldsymbol{w} = \lambda \mathbf{S}_w \boldsymbol{w}}$$
重新写出左边的表达式的定义发现：后两项相乘得到一个常数
$$\mathbf{S}_b \boldsymbol{w} = \Big[ (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^T \Big] \boldsymbol{w}$$
$$\boxed{\mathbf{S}_b \boldsymbol{w} = c (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)}$$
由于我们不关心 $w$ 前面的倍数，最终就得到了：
$$\boldsymbol{w} = \mathbf{S}_w^{-1} (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$$

### 多分类
我们把最后两页拆成 **4 个小关卡**，用“人话”把它们逐一翻译过来。假设现在我们不再只挑“好瓜/坏瓜”（2 分类），而是升级成了 **3 分类问题（$N=3$）：挑“生瓜”、“半熟瓜”、“熟透瓜”**，而且这次每个瓜我们测了 **$d=100$ 个特征**（甜度、重量、颜色、拍打声……）

---

#### 关卡 1：3 类瓜怎么算“类间距离”？（翻译公式 3.40 ~ 3.43）

在 2 分类里，只有 $\boldsymbol{\mu}_0$ 和 $\boldsymbol{\mu}_1$ 两个中心，算类间距离直接一减就行：$\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1$
现在有 **3 类瓜**（3 个中心：$\boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \boldsymbol{\mu}_3$），两两相减太麻烦了，怎么办？

教材想了个很聪明的办法：**找一个“大总管”（所有瓜的共同中心 $\boldsymbol{\mu}$）**！
*   **类间散度 $\mathbf{S}_b$（式 3.43）**：看看每一类瓜的中心 $\boldsymbol{\mu}_i$ 离“总中心 $\boldsymbol{\mu}$”有多远，然后按每类瓜的数量 $m_i$ 加权求和：
    $$\mathbf{S}_b = \sum_{i=1}^{N} m_i (\boldsymbol{\mu}_i - \boldsymbol{\mu})(\boldsymbol{\mu}_i - \boldsymbol{\mu})^T$$
    *人话翻译*：生瓜、半熟瓜、熟透瓜各自的中心，离大部队中心越远，$\mathbf{S}_b$ 就越大（三拨瓜分得越开）。
*   **类内散度 $\mathbf{S}_w$（式 3.41、3.42）**：和二分类一样，就是“生瓜内部的散度 + 半熟瓜内部的散度 + 熟透瓜内部的散度”。
*   **全局散度 $\mathbf{S}_t$（式 3.40）**：不管分类标签，把所有瓜混在一起算总散度。教材顺便告诉你一个性质：$\mathbf{S}_t = \mathbf{S}_w + \mathbf{S}_b$（**总散度 = 类内散度 + 类间散度**）。所以这三个矩阵里，只要挑其中两个（通常就用 $\mathbf{S}_b$ 和 $\mathbf{S}_w$）来列方程就够了，不需要全用。

---

#### 关卡 2：为什么要投影到 $N-1$ 维，而不是一条直线？（理解矩阵 $\mathbf{W}$）

二分类时，我们只找**一条直线** $\boldsymbol{w}$，把样本变成 **1 个数**。
那 3 分类时，为什么教材突然不用向量 $\boldsymbol{w}$，而改用矩阵 $\mathbf{W} \in \mathbb{R}^{d \times (N-1)}$，说要“投影到 $N-1$ 维空间”呢？

##### 1. 为什么 1 条直线不够用了？（几何直觉）
想象一下，生瓜、半熟瓜、熟透瓜在空间里呈**三角形**分布：
*   如果强行把它们全拍扁到 **1 条直线**上，很容易出现“中间那一类被左右两类夹在中间、甚至重叠”的尴尬情况
*   但如果我们把它们拍扁到 **1 张纸（2 维平面）**上，3 个点正好可以在纸上摆成一个漂亮的三角形，彼此互不干扰！

##### 2. 为什么刚好是 $N-1$ 维？
这是几何规律决定的：
*   2 个中心点，最多确定一条 **1 维直线**（$2-1=1$）
*   3 个中心点，最多确定一个 **2 维平面**（$3-1=2$）
*   4 个中心点，最多确定一个 **3 维空间**（$4-1=3$）
*   推广到 $N$ 类：不管原始特征有多少维（就算 $d=100$ 甚至 $10000$），$N$ 个类的中心点**最多只能撑起一个 $N-1$ 维的空间**。再多出的维度里，中心点已经完全扁了，没有任何区分度

##### 3. 神秘的矩阵 $\mathbf{W} \in \mathbb{R}^{d \times (N-1)}$ 到底是什么？
别被“矩阵”吓到！针对我们 3 分类（$N-1 = 2$）、100 个特征（$d = 100$）的西瓜例子，矩阵 $\mathbf{W}$ 其实就是**把 2 个投影方向 $\boldsymbol{w}_1$ 和 $\boldsymbol{w}_2$ 并排站在一起**：
$$\mathbf{W} = \Big[\, \boldsymbol{w}_1 \ , \ \boldsymbol{w}_2 \,\Big] \quad \text{（100 行，2 列）}$$
*   第 1 列 $\boldsymbol{w}_1$ 负责算新坐标的**横坐标** $y_1 = \boldsymbol{w}_1^T \boldsymbol{x}$；
*   第 2 列 $\boldsymbol{w}_2$ 负责算新坐标的**纵坐标** $y_2 = \boldsymbol{w}_2^T \boldsymbol{x}$

于是，一个拥有 100 个特征的复杂西瓜 $\boldsymbol{x}$，经过 $\mathbf{W}^T \boldsymbol{x}$ 一乘，瞬间变成了一张二维白纸上的一个点 $(y_1, y_2)$！这就是**投影矩阵 $\mathbf{W}$** 的真实面目

---

#### 关卡 3：公式 (3.44) 里的 $\text{tr}(\cdot)$ 是什么鬼？

看着公式 (3.44) 很吓人：
$$\max_{\mathbf{W}} \frac{\text{tr}(\mathbf{W}^T \mathbf{S}_b \mathbf{W})}{\text{tr}(\mathbf{W}^T \mathbf{S}_w \mathbf{W})}$$

先认识符号：**$\text{tr}(A)$ 叫矩阵的“迹”，意思极其简单——把方阵主对角线（从左上到右下）的数字加起来！**

为什么要加对角线呢？我们把 $\mathbf{W} = [\boldsymbol{w}_1, \boldsymbol{w}_2]$ 代入分子 $\mathbf{W}^T \mathbf{S}_b \mathbf{W}$ 算一下矩阵乘法，结果是一个 $2 \times 2$ 的小方阵：
$$\mathbf{W}^T \mathbf{S}_b \mathbf{W} = \begin{pmatrix} \boldsymbol{w}_1^T \mathbf{S}_b \boldsymbol{w}_1 & \boldsymbol{w}_1^T \mathbf{S}_b \boldsymbol{w}_2 \\ \boldsymbol{w}_2^T \mathbf{S}_b \boldsymbol{w}_1 & \boldsymbol{w}_2^T \mathbf{S}_b \boldsymbol{w}_2 \end{pmatrix}$$
盯住主对角线（左上和右下）看：
*   左上角 $\boldsymbol{w}_1^T \mathbf{S}_b \boldsymbol{w}_1$：正是**横轴方向 $\boldsymbol{w}_1$ 上的类间散度**！
*   右下角 $\boldsymbol{w}_2^T \mathbf{S}_b \boldsymbol{w}_2$：正是**纵轴方向 $\boldsymbol{w}_2$ 上的类间散度**！

所以，求迹 $\text{tr}(\mathbf{W}^T \mathbf{S}_b \mathbf{W})$ 的结果就是：
$$\text{横轴的类间散度} + \text{纵轴的类间散度} = \textbf{在这张二维纸上的总类间散度！}$$
同理，分母 $\text{tr}(\mathbf{W}^T \mathbf{S}_w \mathbf{W})$ 就是**在这张二维纸上的总类内散度**！

你看，公式 (3.44) 说了半天，**和二分类要做的事情一模一样**：找一张二维纸（由 $\boldsymbol{w}_1, \boldsymbol{w}_2$ 张成），让纸上**总类间散度 / 总类内散度**的比值最大！

---

#### 关卡 4：怎么求解出 $\mathbf{W}$？为什么叫“监督降维”？（翻译公式 3.45 及最后一段）

##### 1. 翻译“广义特征值与特征向量”
公式 (3.45) 写着：$\mathbf{S}_b \mathbf{W} = \lambda \mathbf{S}_w \mathbf{W}$，然后说“$\mathbf{W}$ 的闭式解是 $\mathbf{S}_w^{-1}\mathbf{S}_b$ 的 $N-1$ 个最大广义特征值所对应的特征向量组成的矩阵”

把这句话翻译成大白话就是：
1.  在 100 维空间里，有无数个投影方向 $\boldsymbol{w}$。利用求导（和二分类一样的套路），数学家证明了最优的投影方向必须满足方程：
    $$(\mathbf{S}_w^{-1}\mathbf{S}_b)\boldsymbol{w} = \lambda \boldsymbol{w}$$
    这就是线代里的**特征值方程**
2.  你可以把 $\mathbf{S}_w^{-1}\mathbf{S}_b$ 想象成一个**“方向评分器”**：
    *   它会在 100 维空间里找出 100 个互不干扰的“备选方向”（叫做**特征向量** $\boldsymbol{w}$）；
    *   同时给每个方向打出一个分数（叫做**特征值** $\lambda$）。**分数 $\lambda$ 越高，说明往这个方向投影后，各类西瓜分得越开！**
3.  因为只有 3 类瓜，100 个备选方向里其实最多只有 $3-1=2$ 个方向的分数 $\lambda > 0$（有效信息），剩下 98 个方向的分数全都是 $0$（废话信息）
4.  所以算法怎么做？**用计算机算出所有方向和分数，挑出分数最高的前 $N-1$ 个方向（比如前 2 名 $\boldsymbol{w}_1, \boldsymbol{w}_2$），把它们竖着拼在一起，就成了矩阵 $\mathbf{W}$！**

*(注：因为 $\mathbf{S}_b \boldsymbol{w}_1 = \lambda_1 \mathbf{S}_w \boldsymbol{w}_1$ 和 $\mathbf{S}_b \boldsymbol{w}_2 = \lambda_2 \mathbf{S}_w \boldsymbol{w}_2$ 中的 $\lambda_1, \lambda_2$ 其实是不同的数，教材为了省事把整个矩阵写成了 $\mathbf{S}_b \mathbf{W} = \lambda \mathbf{S}_w \mathbf{W}$，你心里知道它是对每一列分别求特征向量再拼起来就行了)*

##### 2. 翻译最后一段：“监督降维技术”
回顾一下我们刚才干了件什么事：
*   **降维**：原来每个瓜有 $d = 100$ 个特征，数据庞大又复杂；乘以 $\mathbf{W}^T$ 后，每个瓜只剩 $N-1 = 2$ 个特征了（从 100 维降到了 2 维，一下子省掉了 98% 的维度！）。
*   **监督**：在降维找 $\mathbf{W}$ 的过程中，我们用到了每个瓜的**类别标签**（知道谁是生瓜、谁是熟瓜），故意挑了最能把这三类瓜区分开的方向来降维（相比之下，第 10 章的 PCA 降维是不看标签的，叫无监督降维）

所以，LDA 既可以拿来**分类**，也经常被当成一种经典的**有监督降维**工具来使用！

---

#### 一句话总结这 2 页教材
多分类 LDA 就是：**因为 $N$ 类数据的中心最多张成一个 $N-1$ 维空间，所以我们用矩阵 $\mathbf{W}$ 一次性找 $N-1$ 个最佳投影方向；让计算机算出 $\mathbf{S}_w^{-1}\mathbf{S}_b$ 得分最高的前 $N-1$ 个方向拼成 $\mathbf{W}$，就能把几百几千维的数据“拍扁”到 $N-1$ 维，同时让不同类别依然分得清清楚楚**

## 多分类学习
考虑 $N$ 个类别 $C_1,C_2.....C_n$ ，多分类学习的基本思路是将问题拆分成若干个二分类问题

为每个二分类问题训练出一个学习器，最终测试时对学习器的结果进行集成
### 拆分策略
给定数据集 $$D=\{(\boldsymbol{x_1},y_1),(\boldsymbol{x_1},y_1)......(\boldsymbol{x_m},y_m)\},  y_i \in\{C_1,C_2......C_n\} $$
#### 1.一对一（OvO）
将这 $N$ 个类别两两分类配对产生 $N(N-1)/2$ 个二分类任务，测试阶段将新样本交给所有这些学习器产生 $N(N-1)/2$ 个结果，最终结果通过投票产生
#### 2.一对多（OvR）
每次选定一个类作为正例其他类作为反例，训练 $N$ 个学习器
测试时如果仅有一个学习器预测正例，则对应的预测标签作为最终输出；如果有多个预测正例，则选择置信度最大的作为输出结果
#### 3.多对多（MvM）

---

### 一、 来龙去脉：为什么需要 ECOC？

很多经典的分类器（比如你刚学完的**对数几率回归**）天生是个**“二分类器”**——它只能回答“是/否”（比如：是不是好瓜？），没法直接回答“这是苹果、香蕉、橘子还是西瓜？”
要把二分类器用在多分类任务上，教材介绍了三种“拆解”思路：
1.  **OvR（一对其余）**：训练 4 个分类器，分别问“是不是苹果？”“是不是香蕉？”……
2.  **OvO（一对一）**：把 4 种水果两两配对，训练 6 个分类器（“苹果 vs 香蕉”、“苹果 vs 橘子”……），最后投票
3.  **MvM（多对多）**：每次把**若干个类**划为正类，**另外若干个类**划为反类来训练分类器！

**ECOC（纠错输出码）就是 MvM 最著名、最经典的实现方法**

---

### 二、 ECOC 的两步核心玩法（手把手演示）

假设我们要区分 4 种水果：$C_1$（苹果）、$C_2$（香蕉）、$C_3$（橘子）、$C_4$（西瓜）
ECOC 分为**编码（训练阶段）**和**解码（测试阶段）**两步走

#### 第一步：编码（Coding）—— 给每种水果发一张“密码条”

我们设计一个下面这样的**编码矩阵**。这里有 4 行（代表 4 种水果），6 列（代表我们要训练 6 个二分类器 $f_1 \sim f_6$）：

| 类别 | $f_1$ | $f_2$ | $f_3$ | $f_4$ | $f_5$ | $f_6$ | **该类的理想编码** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$C_1$（苹果）** | $+1$ | $+1$ | $+1$ | $+1$ | $-1$ | $-1$ | `[+1, +1, +1, +1, -1, -1]` |
| **$C_2$（香蕉）** | $+1$ | $-1$ | $-1$ | $-1$ | $+1$ | $-1$ | `[+1, -1, -1, -1, +1, -1]` |
| **$C_3$（橘子）** | $-1$ | $+1$ | $-1$ | $-1$ | $-1$ | $+1$ | `[-1, +1, -1, -1, -1, +1]` |
| **$C_4$（西瓜）** | $-1$ | $-1$ | $+1$ | $-1$ | $-1$ | $-1$ | `[-1, -1, +1, -1, -1, -1]` |

**这张表怎么用？（竖着看，一列就是一个分类器！）**
*   **训练 $f_1$（看第 1 列）**：把苹果、香蕉标为正类（$+1$），把橘子、西瓜标为反类（$-1$），喂给一个二分类器去训练。也就是说，$f_1$ 学的是“区分（苹果或香蕉）vs（橘子或西瓜）”
*   **训练 $f_2$（看第 2 列）**：把苹果、橘子标为 $+1$，把香蕉、西瓜标为 $-1$，训练第 2 个分类器
*   ……以此类推，训练出 6 个各有分工的二分类器 $f_1 \sim f_6$

而**横着看**，每种水果就拥有了自己独一无二的 6 位“身份密码条”（如表最右侧所示）

---

#### 第二步：解码（Decoding）—— 见证“纠错”的神奇时刻！

现在来了一个新水果，其实它是一个**真正的苹果（$C_1$）**
我们把它同时放进这 6 个训练好的分类器里去预测。正常情况下，理想输出应该和苹果的密码条 `[+1, +1, +1, +1, -1, -1]` 一模一样

**但是！模型不是神仙，难免会犯错！**
假设这天 $f_2$ 分类器抽风了，本来该输出 $+1$，结果它**错判成了 $-1$**！其他 5 个分类器都发挥正常。于是我们拿到的测试预测编码是：
$$\text{预测编码} = \big[\ +1,\ \mathbf{-1},\ +1,\ +1,\ -1,\ -1\ \big]$$

接下来怎么判定它是哪种水果？**拿这条预测编码，去和表里 4 种水果的理想编码算“距离”，谁最近就猜谁！**

我们数一下“有几位不一样”（这在数学上叫**海明距离，Hamming Distance**）：
*   和 **$C_1$（苹果）** `[+1, +1, +1, +1, -1, -1]` 比：只有第 2 位不同 $\implies$ **距离 = 1**
*   和 **$C_2$（香蕉）** `[+1, -1, -1, -1, +1, -1]` 比：第 3、4、5 位不同 $\implies$ **距离 = 3**
*   和 **$C_3$（橘子）** `[-1, +1, -1, -1, -1, +1]` 比：第 1、2、3、4、6 位不同 $\implies$ **距离 = 5**
*   和 **$C_4$（西瓜）** `[-1, -1, +1, -1, -1, -1]` 比：第 1、4 位不同 $\implies$ **距离 = 2**

**结果出炉：**
距离苹果最近（距离只有 $1$，离其他水果至少为 $2$）！所以系统最终还是果断地输出：**苹果！**

**你看懂为什么它叫“纠错”输出码了吗？**
哪怕其中一个分类器 $f_2$ 犯了错，只要其他分类器足够给力，整体依然能把结果**纠正**回来！如果你用普通的 OvO 或 OvR，一旦关键的分类器出错，往往直接导致平票或者判错；而 ECOC 因为编码之间留出了足够的“安全距离”，天生自带容错能力！

---

### 三、 教材里提到的两个细节补充

对照《西瓜书》第 3.5 节，还有两个名词你一定能看到，我们顺便一并解决：

#### 1. 二元码 vs 三元码（教材图 3.5）
*   **二元码**：就是我们上面用的这种，表里只有 $+1$ 和 $-1$。每个分类器训练时，所有类别的数据都要用上
*   **三元码**：表里除了 $+1$ 和 $-1$，还多了一个 **$0$**（代表**“停用类”**）
    *   比如某一列是 `[+1, -1, 0, 0]`，意思是：训练这个分类器时，**只拿苹果（$+1$）和香蕉（$-1$）的数据来训练，完全忽略橘子和西瓜（$0$）！**
    *   *小彩蛋*：如果每一列都只挑 1 个 $+1$、1 个 $-1$、其余全填 $0$，这不就是前面学过的 **OvO（一对一）** 吗？如果每一列只有 1 个 $+1$、其余全填 $-1$，这不就是 **OvR（一对其余）** 吗？所以 **OvR 和 OvO 其实都是 ECOC 的特殊情况！**

#### 2. 海明距离 vs 欧氏距离
上面解码时我们用“有几位不同”叫**海明距离**，适合分类器直接输出硬邦邦的 $+1$ 或 $-1$。
但在实际算法里（比如对数几率回归），分类器往往先输出一个**连续的实数概率/得分**，比如 $f_1$ 输出 $+0.9$（非常确定是 $+1$），$f_2$ 输出 $-0.1$（其实有点犹豫，勉强算 $-1$）
如果直接一刀切把 $-0.1$ 四舍五入成 $-1$，就丢掉了“它其实很犹豫”的信息。这时候我们改用**欧氏距离**（也就是中学学的两点间距离平方和开根号）：
$$\text{欧氏距离} = \sqrt{\sum_{j=1}^{M} (\text{真实码}_j - \text{预测得分}_j)^2}$$
犹豫的 $-0.1$ 带来的惩罚只比理想值 $+1$ 差了 $1.1^2 = 1.21$，而极其确定的 $+0.9$ 几乎不扣分（$0.1^2=0.01$）。这样解码往往比海明距离更准确！

---

### 四、 怎么设计一张“好”的编码表？（常考知识点）

既然 ECOC 这么好用，是不是随便乱填 $+1$ 和 $-1$ 就行？不是的，教材总结了编码矩阵设计的两个黄金准则：

1.  **横向看（类与类之间）：行距越远越好**
    *   任意两种水果的编码之间，至少要有几个位不同（记最小距离为 $d_{\min}$）
    *   在信息论里有个规律：如果任意两行的海明距离至少是 $d_{\min}$，那么系统最多可以**容忍并纠正 $\lfloor \frac{d_{\min}-1}{2} \rfloor$ 个分类器同时犯错**！
    *   *验算*：上面水果例子里，任意两行最少有 $3$ 位不同（$d_{\min}=3$），代入公式 $\lfloor \frac{3-1}{2} \rfloor = 1$，所以可以保证稳稳纠正 **1 个**分类器的错误！如果编码拉长到 $d_{\min}=5$，甚至能同时抗住 **2 个**分类器犯错！
2.  **纵向看（分类器之间）：列与列越不相关越好**
    *   **绝对不能出现两列完全一样**：否则相当于白白训练了两个一模一样的分类器，犯错的时候一起错，起不到纠错作用
    *   **也不能出现两列正好完全相反**（比如一列 `[+1, +1, -1, -1]`，另一列 `[-1, -1, +1, +1]`）：因为这俩本质上分的是同一拨东西，只是正反标签对调了而已，也是冗余的
    *   **也不能出现一整列全 $+1$ 或全 $-1$**：全是正例没有反例，分类器没法训练
3.  **长度权衡**：
    *   编码越长（列数 $M$ 越多），纠错能力越强；但代价是你需要训练的分类器数量也越多，训练和预测时间都会变长。工程上要在这两者之间做权衡

---

## 类别不平衡问题

前面所有的学习方法都假设各个类别样本数目基本相等，但是如果存在类别不平衡问题呢

如果直接训练学习器，得到的分类器可能毫无实际意义

### 1. 再放缩
使用线性模型对输入进行分类时，其实输出值 $y$ 是在和一个阈值进行比较

例如阈值为0.5，认为 $y>0.5$ 代表正例，反之为反例（可以简单理解为比概率）

这时引入几率：反映了正例的可能性与反例的可能性的比值
$$\frac{y}{1-y}>1：此时认为是正例(1)$$
假设样本中正反例数目不相同
$${m^+}：表示正例；{m^-}：表示反例$$
由于我们通常假设训练集是真实样本总体的无偏采样，因此观测几率就代表了真实几率
$$\frac{y}{1-y}>\frac{m^+}{m^-}：此时认为是正例(2)$$
我们要让分类器进行预测时依据(1)公式，但其实是在执行(2)公式，需要进行放缩处理
$$\frac{y'}{1-y'}=\frac{y}{1-y}\times \frac{m^-}{m^+}$$

### 2.其他方法
再放缩思想简单却并不普适，因为没有办法保证"训练集是真实样本总体的无偏采样"

因此引出其他方法
1. 欠采样：去除某种类别的样本使得两类样本数目基本相等
2. 过采样：增加某种类别的样本使得两类样本数目基本相等
3. 直接进行学习，但是在分类器进行预测时引入再放缩公式