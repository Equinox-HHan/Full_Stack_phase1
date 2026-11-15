# 梯度下降法和牛顿法在对数几率回归的最优化
对数几率回归的损失函数是一个**连续可微的凸函数**，但它没有像线性回归那样的“解析解”（闭式解，即直接通过公式一步算出最优参数），因此我们必须使用**迭代优化算法**来一步步逼近最小值

梯度下降法(一阶方法)牛顿法(二阶方法)正是其中最经典的两类算法

下面我将从**损失函数的化简**出发，一步不跳地推导**梯度**、**梯度下降法**、**海森矩阵**以及**牛顿法**的全部数学细节

---

### 一、 准备工作：确立目标函数与核心导数

为了符号简洁，我们做如下约定（采用增广向量形式，把偏置项 $b$ 吸收进权重向量中）：
*   样本数 $m$，原始特征维度 $d$
*   第 $i$ 个样本的增广特征向量：$\boldsymbol{x}_i = (x_{i1}, x_{i2}, \dots, x_{id}, 1)^T \in \mathbb{R}^{d+1}$
*   待求参数向量：$\boldsymbol{\beta} = (w_1, w_2, \dots, w_d, b)^T \in \mathbb{R}^{d+1}$
*   样本真实标签：$y_i \in \{0, 1\}$
*   线性组合：记 $z_i = \boldsymbol{\beta}^T \boldsymbol{x}_i = \sum_{k=1}^{d+1} \beta_k x_{ik}$

#### 1. Sigmoid 函数的求导性质
模型预测样本为正例的概率为：
$$p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}} = (1 + e^{-z_i})^{-1}$$

我们先推导它的导数 $\sigma'(z)$，后面会反复用到：
$$\begin{aligned}
\sigma'(z) &= \frac{d}{dz}(1 + e^{-z})^{-1} \\
&= -(1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z}) \\
&= -(1 + e^{-z})^{-2} \cdot (-e^{-z}) \\
&= \frac{e^{-z}}{(1 + e^{-z})^2} \\
&= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} \\
&= \frac{1}{1 + e^{-z}} \cdot \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} \\
&= \frac{1}{1 + e^{-z}} \cdot \left(1 - \frac{1}{1 + e^{-z}}\right) \\
&= \sigma(z)\big(1 - \sigma(z)\big)
\end{aligned}$$
记住这个优美的结论：**$\sigma'(z) = \sigma(z)(1 - \sigma(z))$**，即 $\frac{dp_i}{dz_i} = p_i(1 - p_i)$

#### 2. 损失函数的化简
由极大似然估计（或交叉熵）得到的对数几率回归损失函数为：
$$J(\boldsymbol{\beta}) = -\frac{1}{m} \sum_{i=1}^m \Big[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \Big]$$
为了求导方便，我们先把中括号里的式子化简：
$$\begin{aligned}
y_i \ln p_i + (1 - y_i) \ln(1 - p_i) &= y_i \ln p_i + \ln(1 - p_i) - y_i \ln(1 - p_i) \\
&= y_i \ln \frac{p_i}{1 - p_i} + \ln(1 - p_i)
\end{aligned}$$
把 $p_i = \frac{1}{1 + e^{-z_i}}$ 代入这两项：
*   $\frac{p_i}{1 - p_i} = \frac{\frac{1}{1+e^{-z_i}}}{\frac{e^{-z_i}}{1+e^{-z_i}}} = e^{z_i}$，所以 $\ln \frac{p_i}{1 - p_i} = z_i$。
*   $1 - p_i = \frac{e^{-z_i}}{1 + e^{-z_i}} = \frac{1}{e^{z_i} + 1}$，所以 $\ln(1 - p_i) = -\ln(1 + e^{z_i})$

代回原式（注意外面的负号）：
$$J(\boldsymbol{\beta}) = \frac{1}{m} \sum_{i=1}^m \Big[ \ln(1 + e^{\boldsymbol{\beta}^T \boldsymbol{x}_i}) - y_i \boldsymbol{\beta}^T \boldsymbol{x}_i \Big]$$
这就是我们要**最小化**的目标函数

---

### 二、 梯度的严格推导（两种方法都要用）

梯度 $\nabla J(\boldsymbol{\beta})$ 是一个由所有一阶偏导数组成的列向量：
$$\nabla J(\boldsymbol{\beta}) = \left( \frac{\partial J}{\partial \beta_1}, \frac{\partial J}{\partial \beta_2}, \dots, \frac{\partial J}{\partial \beta_{d+1}} \right)^T$$

为了不跳步，我们先对**单个分量 $\beta_j$** 求偏导。已知 $z_i = \sum_{k=1}^{d+1} \beta_k x_{ik}$，显然有 $\frac{\partial z_i}{\partial \beta_j} = x_{ij}$

利用链式法则对 $J(\boldsymbol{\beta})$ 求 $\beta_j$ 的偏导：
$$\begin{aligned}
\frac{\partial J(\boldsymbol{\beta})}{\partial \beta_j} &= \frac{1}{m} \sum_{i=1}^m \left[ \frac{\partial}{\partial \beta_j}\ln(1 + e^{z_i}) - \frac{\partial}{\partial \beta_j}(y_i z_i) \right] \\
&= \frac{1}{m} \sum_{i=1}^m \left[ \frac{1}{1 + e^{z_i}} \cdot e^{z_i} \cdot \frac{\partial z_i}{\partial \beta_j} - y_i \frac{\partial z_i}{\partial \beta_j} \right]
\end{aligned}$$
注意看 $\frac{e^{z_i}}{1 + e^{z_i}}$，分子分母同除以 $e^{z_i}$，它就是 $\frac{1}{1 + e^{-z_i}} = p_i$！于是上式变为：
$$\frac{\partial J(\boldsymbol{\beta})}{\partial \beta_j} = \frac{1}{m} \sum_{i=1}^m (p_i - y_i) x_{ij}$$

现在把 $j = 1, 2, \dots, d+1$ 全部竖着拼成一个向量，就得到了**梯度向量**：
$$\nabla J(\boldsymbol{\beta}) = \frac{1}{m} \sum_{i=1}^m (p_i - y_i) \boldsymbol{x}_i$$

还可以进一步写成**矩阵形式**（方便代码向量化）：
令数据矩阵 $X \in \mathbb{R}^{m \times (d+1)}$（第 $i$ 行为 $\boldsymbol{x}_i^T$），预测向量 $\boldsymbol{p} = (p_1, \dots, p_m)^T$，标签向量 $\boldsymbol{y} = (y_1, \dots, y_m)^T$
检查矩阵乘法 $X^T (\boldsymbol{p} - \boldsymbol{y})$ 的第 $j$ 个元素：它是 $X^T$ 的第 $j$ 行（即所有样本的第 $j$ 维特征 $x_{1j}, \dots, x_{mj}$）与列向量 $(\boldsymbol{p} - \boldsymbol{y})$ 的内积，正好等于 $\sum_{i=1}^m (p_i - y_i) x_{ij}$。因此：
$$\nabla J(\boldsymbol{\beta}) = \frac{1}{m} X^T (\boldsymbol{p} - \boldsymbol{y})$$

---

### 三、 梯度下降法（Gradient Descent）

#### 1. 数学原理：为什么往“负梯度”方向走？
很多人直觉知道负梯度是下降最快的方向，这背后的严格数学依据是**一阶泰勒展开**和**柯西-施瓦茨不等式**

假设当前参数位于 $\boldsymbol{\beta}$，我们想移动一个很小的步长 $\Delta \boldsymbol{\beta} = \eta \boldsymbol{u}$，其中 $\eta > 0$ 是非常小的正数，$\boldsymbol{u}$ 是代表移动方向的**单位向量**（即 $\|\boldsymbol{u}\|_2 = 1$）

把 $J(\boldsymbol{\beta} + \Delta \boldsymbol{\beta})$ 在 $\boldsymbol{\beta}$ 处做一阶泰勒展开（忽略高阶无穷小）：
$$J(\boldsymbol{\beta} + \eta \boldsymbol{u}) \approx J(\boldsymbol{\beta}) + \nabla J(\boldsymbol{\beta})^T (\eta \boldsymbol{u}) = J(\boldsymbol{\beta}) + \eta \big(\nabla J(\boldsymbol{\beta})^T \boldsymbol{u}\big)$$

我们的目标是让新位置的函数值 $J(\boldsymbol{\beta} + \eta \boldsymbol{u})$ 比原来尽可能小，也就是让变化量 $\eta \big(\nabla J(\boldsymbol{\beta})^T \boldsymbol{u}\big)$ **尽可能为负且绝对值最大**

利用向量的内积公式（或柯西-施瓦茨不等式）：
$$\nabla J(\boldsymbol{\beta})^T \boldsymbol{u} = \|\nabla J(\boldsymbol{\beta})\|_2 \cdot \|\boldsymbol{u}\|_2 \cdot \cos\theta$$
其中 $\theta$ 是梯度 $\nabla J(\boldsymbol{\beta})$ 与方向 $\boldsymbol{u}$ 之间的夹角。因为 $\|\boldsymbol{u}\|_2 = 1$ 且 $\|\nabla J(\boldsymbol{\beta})\|_2 \ge 0$，要使内积最小（最负），必须让 $\cos\theta = -1$，即 $\theta = 180^\circ$

这意味着，**方向 $\boldsymbol{u}$ 必须与梯度 $\nabla J(\boldsymbol{\beta})$ 完全相反**：
$$\boldsymbol{u} = -\frac{\nabla J(\boldsymbol{\beta})}{\|\nabla J(\boldsymbol{\beta})\|_2}$$

#### 2. 更新公式
在实际算法中，我们把常数分母 $\|\nabla J(\boldsymbol{\beta})\|_2$ 直接吸收到步长（学习率） $\eta$ 中，得到：
$$\Delta \boldsymbol{\beta} = -\eta \nabla J(\boldsymbol{\beta})$$
*注：保留梯度模长 $\|\nabla J(\boldsymbol{\beta})\|_2$ 还有个额外好处——越靠近极小值点，梯度越小，迈出的步子就会自动变小，不容易冲过头*

将前面推导的对数几率回归梯度代入，得到第 $t+1$ 步的迭代更新公式：
$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta \cdot \frac{1}{m} \sum_{i=1}^m \Big( \sigma\big({\boldsymbol{\beta}^{(t)}}^T \boldsymbol{x}_i\big) - y_i \Big) \boldsymbol{x}_i$$
或矩阵形式：
$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \frac{\eta}{m} X^T (\boldsymbol{p}^{(t)} - \boldsymbol{y})$$

#### 3. 局限性
1.  **超参数敏感**：学习率 $\eta$ 需要手动调节。太小则收敛极慢；太大则可能在谷底来回震荡甚至发散
2.  **“之”字形震荡（病态曲率问题）**：梯度下降只看当前位置的一阶斜率，不知道各方向弯曲程度（曲率）的差异，如果损失函数的等高线是一个狭长的椭圆（常见于特征尺度差异大时），负梯度方向并不指向椭圆中心，导致路线呈“之”字形前进，收敛慢（线性收敛）

---

### 四、 牛顿法（Newton's Method）

为了解决梯度下降“看不到曲面弯曲程度”的缺点，牛顿法引入了**二阶导数**，用局部的**二次曲面（抛物面）**去逼近目标函数，然后一步直接跳到这个二次曲面的最低点！

#### 1. 数学原理：二阶泰勒展开与二次近似极小化
将 $J(\boldsymbol{\beta} + \Delta \boldsymbol{\beta})$ 在当前点 $\boldsymbol{\beta}$ 处做**二阶泰勒展开**：
$$J(\boldsymbol{\beta} + \Delta \boldsymbol{\beta}) \approx q(\Delta \boldsymbol{\beta}) = J(\boldsymbol{\beta}) + \boldsymbol{g}^T \Delta \boldsymbol{\beta} + \frac{1}{2} \Delta \boldsymbol{\beta}^T H \Delta \boldsymbol{\beta}$$
其中：
*   $\boldsymbol{g} = \nabla J(\boldsymbol{\beta}) \in \mathbb{R}^{d+1}$ 是当前点的梯度向量
*   $H = \nabla^2 J(\boldsymbol{\beta}) \in \mathbb{R}^{(d+1) \times (d+1)}$ 是当前点的**海森矩阵（Hessian Matrix）**，其第 $j$ 行第 $k$ 列元素定义为二阶偏导数：
    $$H_{jk} = \frac{\partial^2 J(\boldsymbol{\beta})}{\partial \beta_j \partial \beta_k}$$
    （因为 $J$ 二阶连续可微，混合偏导与求导顺序无关，所以 $H_{jk} = H_{kj}$，即 $H$ 是对称矩阵）

现在，我们把 $q(\Delta \boldsymbol{\beta})$ 看作关于步长向量 $\Delta \boldsymbol{\beta}$ 的函数。为了找到能使近似函数 $q(\Delta \boldsymbol{\beta})$ 最小的 $\Delta \boldsymbol{\beta}$，我们对 $q(\Delta \boldsymbol{\beta})$ 关于 $\Delta \boldsymbol{\beta}$ 求梯度并令其为 $\mathbf{0}$

为了不跳步，我们按分量 $\Delta \beta_k$ 展开求导（记 $\boldsymbol{v} = \Delta \boldsymbol{\beta}$）：
*   第一项 $J(\boldsymbol{\beta})$ 与 $\boldsymbol{v}$ 无关，导数为 $0$。
*   第二项 $\boldsymbol{g}^T \boldsymbol{v} = \sum_{a} g_a v_a$，对 $v_k$ 求偏导得 $g_k$。拼成向量就是 $\boldsymbol{g}$
*   第三项 $\frac{1}{2} \boldsymbol{v}^T H \boldsymbol{v} = \frac{1}{2} \sum_{a} \sum_{b} H_{ab} v_a v_b$。对 $v_k$ 求偏导时，只有当 $a=k$ 或 $b=k$（或两者都为 $k$）时才不为 $0$（利用乘法求导法则）：
    $$\frac{\partial}{\partial v_k}\left(\frac{1}{2} \sum_{a,b} H_{ab} v_a v_b\right) = \frac{1}{2}\left( \sum_{b} H_{kb} v_b + \sum_{a} H_{ak} v_a \right)$$
    因为 $H$ 是对称矩阵（$H_{ak} = H_{ka}$），所以括号里两项完全相等，合并后消去 $\frac{1}{2}$：
    $$= \sum_{b} H_{kb} v_b$$
    这正好是矩阵乘向量 $H\boldsymbol{v}$ 的第 $k$ 个分量！拼成向量就是 $H \Delta \boldsymbol{\beta}$

综上，令梯度等于零向量：
$$\nabla_{\Delta \boldsymbol{\beta}} q(\Delta \boldsymbol{\beta}) = \boldsymbol{g} + H \Delta \boldsymbol{\beta} = \mathbf{0}$$
当 $H$ 可逆（正定）时，解得最佳移动步长（称为**牛顿方向/牛顿步**）：
$$\Delta \boldsymbol{\beta} = -H^{-1} \boldsymbol{g}$$

*(注：牛顿法也可以从“求一阶导数 $\nabla J(\boldsymbol{\beta}) = \mathbf{0}$ 的根”的角度推导：对 $\nabla J(\boldsymbol{\beta} + \Delta \boldsymbol{\beta}) \approx \nabla J(\boldsymbol{\beta}) + H \Delta \boldsymbol{\beta} = \mathbf{0}$，同样直接得出 $\Delta \boldsymbol{\beta} = -H^{-1} \boldsymbol{g}$)*

#### 2. 推导对数几率回归的 Hessian 矩阵
刚才我们已经算出了一阶偏导：
$$g_j = \frac{\partial J(\boldsymbol{\beta})}{\partial \beta_j} = \frac{1}{m} \sum_{i=1}^m (p_i - y_i) x_{ij}$$
现在再对 $\beta_k$ 求一次偏导，计算 $H_{jk}$：
$$H_{jk} = \frac{\partial g_j}{\partial \beta_k} = \frac{1}{m} \sum_{i=1}^m x_{ij} \frac{\partial (p_i - y_i)}{\partial \beta_k} = \frac{1}{m} \sum_{i=1}^m x_{ij} \frac{\partial p_i}{\partial \beta_k}$$
由于 $y_i$ 是常数导数为 $0$，而 $p_i = \sigma(z_i)$，利用第一章推导出的 Sigmoid 求导公式和链式法则：
$$\frac{\partial p_i}{\partial \beta_k} = \sigma'(z_i) \cdot \frac{\partial z_i}{\partial \beta_k} = p_i(1 - p_i) x_{ik}$$
代回 $H_{jk}$：
$$H_{jk} = \frac{1}{m} \sum_{i=1}^m x_{ij} \cdot p_i(1 - p_i) \cdot x_{ik}$$

把它写成极其简洁的**矩阵形式**：
定义一个 $m \times m$ 的对角矩阵 $D$，其对角线元素为 $D_{ii} = p_i(1 - p_i)$：
$$D = \begin{pmatrix} p_1(1-p_1) & 0 & \cdots & 0 \\ 0 & p_2(1-p_2) & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & p_m(1-p_m) \end{pmatrix}$$
我们来验证一下 $X^T D X$ 的第 $j$ 行第 $k$ 列元素：
$$(X^T D X)_{jk} = \sum_{i=1}^m (X^T)_{ji} (DX)_{ik} = \sum_{i=1}^m x_{ij} \big( D_{ii} x_{ik} \big) = \sum_{i=1}^m x_{ij} p_i(1 - p_i) x_{ik}$$
完全一致！因此，对数几率回归的 Hessian 矩阵为：
$$H = \frac{1}{m} X^T D X$$

#### 3. 顺便证明损失函数的凸性
有了 $H = \frac{1}{m} X^T D X$，我们可以顺便证明 $J(\boldsymbol{\beta})$ 一定是**凸函数**（没有局部极小值陷阱，找到的就是全局最小值）
对任意非零向量 $\boldsymbol{v} \in \mathbb{R}^{d+1}$：
$$\boldsymbol{v}^T H \boldsymbol{v} = \frac{1}{m} \boldsymbol{v}^T X^T D X \boldsymbol{v} = \frac{1}{m} (X\boldsymbol{v})^T D (X\boldsymbol{v}) = \frac{1}{m} \sum_{i=1}^m p_i(1 - p_i) \big((X\boldsymbol{v})_i\big)^2$$
因为 $p_i \in (0, 1)$，所以 $p_i(1 - p_i) > 0$，而平方项 $\big((X\boldsymbol{v})_i\big)^2 \ge 0$。因此恒有 $\boldsymbol{v}^T H \boldsymbol{v} \ge 0$，即 **$H$ 处处半正定**（若 $X$ 列满秩，则 $H$ 严格正定）。这就从数学上保证了 $J(\boldsymbol{\beta})$ 是凸函数

#### 4. 牛顿法更新公式
将梯度 $\boldsymbol{g} = \frac{1}{m} X^T (\boldsymbol{p} - \boldsymbol{y})$ 和海森矩阵 $H = \frac{1}{m} X^T D X$ 代入 $\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - H^{-1} \boldsymbol{g}$：
$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \left( X^T D^{(t)} X \right)^{-1} X^T (\boldsymbol{p}^{(t)} - \boldsymbol{y})$$
*(注意：梯度和牛顿矩阵前面的常数 $\frac{1}{m}$ 在 $H^{-1} \boldsymbol{g}$ 中正好互相抵消了！)*

**几何意义与优势**：
牛顿法不需要像梯度下降那样预设学习率 $\eta$。$H^{-1}$ 本身就包含了所有方向上的曲率信息，在平缓方向上步子迈得大，在陡峭方向上步子迈得小，本质上相当于把椭圆等高线通过线性变换“掰圆”了，然后直奔中心。在接近最优点时，它具有**二阶（二次）收敛速度**（例如误差每次迭代从 $0.01 \to 0.0001 \to 10^{-8}$），通常十几步就能收敛

---

### 五、 梯度下降 vs 牛顿法：对比与工程实践

| 对比维度 | 梯度下降法 (Gradient Descent) | 牛顿法 (Newton's Method) |
| :--- | :--- | :--- |
| **数学阶数** | 一阶优化算法（仅用梯度 $\boldsymbol{g}$） | 二阶优化算法（用梯度 $\boldsymbol{g}$ + 海森矩阵 $H$） |
| **步长/学习率** | 需要人为设置超参数 $\eta$ | 理论上不需要（阻尼牛顿法会配合线搜索） |
| **收敛速度** | 慢（线性收敛，受制于条件数/特征缩放） | 极快（局部二次收敛） |
| **单步计算复杂度** | 极低：$O(md)$ | 极高：$O(md^2 + d^3)$ |
| **内存占用** | $O(d)$ | $O(d^2)$（需要存储海森矩阵 $H$） |
| **适用场景** | **样本量大或特征维度 $d$ 很高**（如深度学习、百万维稀疏特征） | **特征维度 $d$ 较小**（如 $d < 1000$ 的传统统计模型） |

#### 牛顿法在实战中的两个痛点与改进：
1.  **计算瓶颈（$O(d^3)$ 求逆）**：如果特征维度 $d = 10000$，海森矩阵有 $1$ 亿个元素，求一次逆矩阵需要 $10^{12}$ 次运算，完全不可接受。因此工程上衍生出了**拟牛顿法（Quasi-Newton Methods，如 BFGS 和 L-BFGS）**——它们不直接计算和求逆海森矩阵，而是利用前后几步的梯度变化去“近似” $H^{-1}$，既保留了接近牛顿法的快速度，又把复杂度降到了 $O(d^2)$ 甚至 $O(md)$
2.  **$H$ 奇异（不可逆）问题**：
    *   如果样本特征存在多重共线性（$X$ 不满秩），或者在迭代后期某些样本预测概率 $p_i$ 极其接近 $0$ 或 $1$（数据线性可分时常发生），导致计算机浮点数下溢使得 $p_i(1 - p_i) = 0$，此时 $X^T D X$ 就会不可逆
    *   **解决办法**：在损失函数中加入 **L2 正则化项** $\frac{\lambda}{2}\|\boldsymbol{\beta}\|^2$。加入后，海森矩阵变为 $H_{reg} = \frac{1}{m}X^T D X + \lambda I$，特征值全部至少大于 $\lambda > 0$，矩阵就变得**严格正定且必定可逆**了

---

**总结一下核心脉络**：
*   **梯度下降**是用**切平面**近似局部函数，沿着斜率最陡的负梯度走小步：$\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - \eta \boldsymbol{g}$
*   **牛顿法**是用**二次抛物面**近似局部函数，直接跳到抛物面的最低点：$\boldsymbol{\beta} \leftarrow \boldsymbol{\beta} - H^{-1} \boldsymbol{g}$

《西瓜书》这一节（3.4 线性判别分析，简称 LDA）写得非常凝练，很多线性代数和微积分的步骤都“一笔带过”了，基础薄弱的同学看这部分确实容易卡住。

下面我**严格对照你发的这 4 页教材**，把每一个跳跃的公式掰开揉碎，在需要用到概率论和线性代数的地方直接给你做“前置小补给”，带你顺畅地走完整个推导！

---

# LDA内容
### 第一部分：核心思想与“投影”的数学表达（对应第 1 页）

#### 1. 直观想法
看图 3.3：二维平面上有两堆点（正例 `+` 和反例 `-`）。LDA 想找一条过原点的直线（方向由向量 $\boldsymbol{w}$ 决定），把所有二维的点都“拍扁”投影到这条一维直线上
好的直线要满足两个要求：
*   **类内尽可能近**：同一类的点投影后，紧挨在一起（方差小）
*   **类间尽可能远**：两类点的中心（红圆和红三角）投影后，离得越远越好

#### 2. 【线代小补给】为什么投影是 $y = \boldsymbol{w}^T \boldsymbol{x}$？
设样本是列向量 $\boldsymbol{x} = (x_1, x_2)^T$，直线的方向也是列向量 $\boldsymbol{w} = (w_1, w_2)^T$
在中学几何里，向量 $\boldsymbol{x}$ 在 $\boldsymbol{w}$ 方向上的投影长度是：$\|\boldsymbol{x}\| \cos\theta$
而两个向量的**内积（点乘）**定义为：
$$\boldsymbol{w}^T \boldsymbol{x} = w_1 x_1 + w_2 x_2 = \|\boldsymbol{w}\| \cdot \|\boldsymbol{x}\| \cos\theta$$
你看，$\boldsymbol{w}^T \boldsymbol{x}$ 就是投影长度再乘上常数 $\|\boldsymbol{w}\|$。因为所有点都乘以同一个常数 $\|\boldsymbol{w}\|$（相当于把直线上刻度的尺子等比例拉长或缩短），所以不影响点的相对位置和聚散程度。为了计算方便，教材直接把样本 $\boldsymbol{x}$ 的投影位置记为实数 $\boldsymbol{w}^T \boldsymbol{x}$

#### 3. 【概率与线代小补给】均值向量 $\boldsymbol{\mu}_i$ 和协方差矩阵 $\boldsymbol{\Sigma}_i$
教材第一页底部给出了 $X_i$、$\boldsymbol{\mu}_i$、$\boldsymbol{\Sigma}_i$。假设第 $0$ 类有 $m_0$ 个样本，第 $1$ 类有 $m_1$ 个样本：
*   **均值向量 $\boldsymbol{\mu}_i$**：就是把第 $i$ 类所有样本向量加起来求平均（中心点）例如：
    $$\boldsymbol{\mu}_0 = \frac{1}{m_0} \sum_{\boldsymbol{x} \in X_0} \boldsymbol{x}$$
*   **协方差矩阵（散度矩阵）$\boldsymbol{\Sigma}_i$**：注意！在公式 (3.33) 中，教材把 $\boldsymbol{\Sigma}_0$ 写成了 $\sum_{\boldsymbol{x}\in X_0} (\boldsymbol{x} - \boldsymbol{\mu}_0)(\boldsymbol{x} - \boldsymbol{\mu}_0)^T$。
    *   *小提醒*：严格的概率论里协方差矩阵前面还要除以样本数 $\frac{1}{m_0}$ 或 $\frac{1}{m_0-1}$，教材为了后面公式简洁把分母省掉了（这种不除以样本数的矩阵其实叫“散度矩阵”，用来衡量数据的分散程度）
    *   *矩阵维度怎么算？* $\boldsymbol{x} - \boldsymbol{\mu}_0$ 是一个 $d \times 1$ 的列向量，它的转置 $(\boldsymbol{x} - \boldsymbol{\mu}_0)^T$ 是 $1 \times d$ 的行向量。**列向量 $\times$ 行向量 = 一个 $d \times d$ 的方阵**（这叫向量的“外积”），它描述了样本各个特征维度自身的方差以及维度之间的相关性

#### 4. 关键推导：为什么投影后的中心是 $\boldsymbol{w}^T \boldsymbol{\mu}_i$，协方差（方差）是 $\boldsymbol{w}^T \boldsymbol{\Sigma}_i \boldsymbol{w}$？
教材直接写出了这两个结论，我们一步步算出来看：
1.  **投影后的中心**：把第 $0$ 类每个样本投影，得到一堆一维实数 $\boldsymbol{w}^T \boldsymbol{x}$，求它们的平均值：
    $$\text{投影中心} = \frac{1}{m_0} \sum_{\boldsymbol{x} \in X_0} (\boldsymbol{w}^T \boldsymbol{x}) = \boldsymbol{w}^T \left( \frac{1}{m_0} \sum_{\boldsymbol{x} \in X_0} \boldsymbol{x} \right) = \boldsymbol{w}^T \boldsymbol{\mu}_0$$
    （就是一个实数！）

2.  **投影后的分散程度（方差/散度）**：在一维直线上，衡量一堆数分散程度的指标是“每个数减去平均值后的平方和”：
    $$\text{投影散度} = \sum_{\boldsymbol{x} \in X_0} (\boldsymbol{w}^T \boldsymbol{x} - \boldsymbol{w}^T \boldsymbol{\mu}_0)^2 = \sum_{\boldsymbol{x} \in X_0} \big[ \boldsymbol{w}^T (\boldsymbol{x} - \boldsymbol{\mu}_0) \big]^2$$
    **这里有一个超级常用的线代小技巧**：括号里的 $a = \boldsymbol{w}^T (\boldsymbol{x} - \boldsymbol{\mu}_0)$ 是一个**实数**，实数的转置等于它自己，即 $a = a^T$！
    根据矩阵转置法则 $(AB)^T = B^T A^T$，有：
    $$a^T = \big[ \boldsymbol{w}^T (\boldsymbol{x} - \boldsymbol{\mu}_0) \big]^T = (\boldsymbol{x} - \boldsymbol{\mu}_0)^T (\boldsymbol{w}^T)^T = (\boldsymbol{x} - \boldsymbol{\mu}_0)^T \boldsymbol{w}$$
    于是平方 $a^2 = a \cdot a^T$ 可以写成：
    $$\big[ \boldsymbol{w}^T (\boldsymbol{x} - \boldsymbol{\mu}_0) \big]^2 = \boldsymbol{w}^T (\boldsymbol{x} - \boldsymbol{\mu}_0) (\boldsymbol{x} - \boldsymbol{\mu}_0)^T \boldsymbol{w}$$
    把 $\boldsymbol{w}^T$ 和 $\boldsymbol{w}$ 从求和号 $\sum$ 中提出来：
    $$\text{投影散度} = \boldsymbol{w}^T \left[ \sum_{\boldsymbol{x} \in X_0} (\boldsymbol{x} - \boldsymbol{\mu}_0) (\boldsymbol{x} - \boldsymbol{\mu}_0)^T \right] \boldsymbol{w} = \boldsymbol{w}^T \boldsymbol{\Sigma}_0 \boldsymbol{w}$$
    （结果同样是一个实数！）

---

### 第二部分：目标函数与拉格朗日求解（对应第 2 页）

#### 1. 理解公式 (3.32) 到 (3.35)
我们要让“类间距离”尽量大，“类内散度”尽量小，最自然的想法就是让两者的**比值**最大：
$$J = \frac{\|\boldsymbol{w}^T \boldsymbol{\mu}_0 - \boldsymbol{w}^T \boldsymbol{\mu}_1\|_2^2}{\boldsymbol{w}^T \boldsymbol{\Sigma}_0 \boldsymbol{w} + \boldsymbol{w}^T \boldsymbol{\Sigma}_1 \boldsymbol{w}}$$
*   **分子怎么变的？** 因为 $\boldsymbol{w}^T \boldsymbol{\mu}_0 - \boldsymbol{w}^T \boldsymbol{\mu}_1 = \boldsymbol{w}^T (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$ 是一个一维实数，实数的 $L_2$ 范数平方 $\|\cdot\|_2^2$ 就是它自己的平方！用刚才一模一样的 $a^2 = a \cdot a^T$ 技巧：
    $$\|\boldsymbol{w}^T (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)\|_2^2 = \boldsymbol{w}^T (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^T \boldsymbol{w}$$
*   **分母怎么变的？** 用分配律提取公因式：$\boldsymbol{w}^T \boldsymbol{\Sigma}_0 \boldsymbol{w} + \boldsymbol{w}^T \boldsymbol{\Sigma}_1 \boldsymbol{w} = \boldsymbol{w}^T (\boldsymbol{\Sigma}_0 + \boldsymbol{\Sigma}_1) \boldsymbol{w}$

接着教材给分子中间和分母中间的 $d \times d$ 矩阵起了个名字：
*   **类内散度矩阵** $\mathbf{S}_w = \boldsymbol{\Sigma}_0 + \boldsymbol{\Sigma}_1$ —— 公式 (3.33)
*   **类间散度矩阵** $\mathbf{S}_b = (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^T$ —— 公式 (3.34)
    *(提示：这两个矩阵都是**对称矩阵**，即 $\mathbf{S}_w^T = \mathbf{S}_w$，$\mathbf{S}_b^T = \mathbf{S}_b$，后面求导会用到！)*

目标函数就简化成了漂亮的**公式 (3.35)**：
$$J = \frac{\boldsymbol{w}^T \mathbf{S}_b \boldsymbol{w}}{\boldsymbol{w}^T \mathbf{S}_w \boldsymbol{w}}$$

#### 2. 为什么可以令分母 $\boldsymbol{w}^T \mathbf{S}_w \boldsymbol{w} = 1$？（对应公式 3.36）
你看旁注：“若 $\boldsymbol{w}$ 是一个解，则对任意常数 $\alpha$，$\alpha \boldsymbol{w}$ 也是解”
我们来验算一下，把 $\alpha \boldsymbol{w}$ 代入公式 (3.35)：
$$\frac{(\alpha \boldsymbol{w})^T \mathbf{S}_b (\alpha \boldsymbol{w})}{(\alpha \boldsymbol{w})^T \mathbf{S}_w (\alpha \boldsymbol{w})} = \frac{\alpha^2 (\boldsymbol{w}^T \mathbf{S}_b \boldsymbol{w})}{\alpha^2 (\boldsymbol{w}^T \mathbf{S}_w \boldsymbol{w})} = \frac{\boldsymbol{w}^T \mathbf{S}_b \boldsymbol{w}}{\boldsymbol{w}^T \mathbf{S}_w \boldsymbol{w}}$$
$\alpha^2$ 约掉了！这说明：**$J$ 的大小只跟 $\boldsymbol{w}$ 的方向有关，跟 $\boldsymbol{w}$ 的长短毫无关系！**
既然长短随便选，如果直接求分式的最大值，解会有无穷多个（差一个倍数）为了定下唯一的长度，我们就人为规定一个约束条件：让分母恰好等于 $1$（即 $\boldsymbol{w}^T \mathbf{S}_w \boldsymbol{w} = 1$），然后最大化分子 $\boldsymbol{w}^T \mathbf{S}_b \boldsymbol{w}$（等价于最小化 $-\boldsymbol{w}^T \mathbf{S}_b \boldsymbol{w}$），这就是**公式 (3.36)**

#### 3. 【高数与线代补给】从 (3.36) 到 (3.37)：拉格朗日乘子法是怎么算的？
教材直接写“由拉格朗日乘子法，上式等价于 $\mathbf{S}_b \boldsymbol{w} = \lambda \mathbf{S}_w \boldsymbol{w}$”，这里跳了整整 4 步，我们补全它：

1.  **构造拉格朗日函数**（把等式约束用乘子 $\lambda$ 融合进目标函数）：
    $$L(\boldsymbol{w}, \lambda) = -\boldsymbol{w}^T \mathbf{S}_b \boldsymbol{w} + \lambda (\boldsymbol{w}^T \mathbf{S}_w \boldsymbol{w} - 1)$$
2.  **对向量 $\boldsymbol{w}$ 求偏导（梯度）并令其等于 $\mathbf{0}$**：
    *   *求导公式*：对于任意对称矩阵 $A$（$A^T = A$），二次型函数 $f(\boldsymbol{w}) = \boldsymbol{w}^T A \boldsymbol{w}$ 对向量 $\boldsymbol{w}$ 的导数是：
        $$\frac{\partial (\boldsymbol{w}^T A \boldsymbol{w})}{\partial \boldsymbol{w}} = 2 A \boldsymbol{w}$$
        *(类比标量求导：$(a w^2)' = 2 a w$，完全一样！)*
    *   因为 $\mathbf{S}_b$ 和 $\mathbf{S}_w$ 都是对称矩阵，直接套用公式：
        $$\frac{\partial L}{\partial \boldsymbol{w}} = -2\mathbf{S}_b \boldsymbol{w} + 2\lambda \mathbf{S}_w \boldsymbol{w} = \mathbf{0}$$
3.  移项并消去两边公因子 $2$，立刻得到**公式 (3.37)**：
    $$\mathbf{S}_b \boldsymbol{w} = \lambda \mathbf{S}_w \boldsymbol{w}$$

---

### 第三部分：闭式解的妙招与多分类推广（对应第 3 页）

#### 1. 最神奇的一步：从 (3.37) 到 (3.39)
很多同学看不明白：为什么突然就能“不妨令 $\mathbf{S}_b \boldsymbol{w} = \lambda (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$”？

我们把 $\mathbf{S}_b$ 的定义（3.34）代进去展开看：
$$\mathbf{S}_b \boldsymbol{w} = \Big[ (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^T \Big] \boldsymbol{w}$$
利用矩阵乘法的结合律，把括号改到后面两个向量上：
$$\mathbf{S}_b \boldsymbol{w} = (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1) \cdot \Big[ (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^T \boldsymbol{w} \Big]$$
睁大眼睛看后面的中括号 $(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)^T \boldsymbol{w}$：一个行向量 $\times$ 一个列向量 = **一个普通的实数**！我们把这个实数记为 $c$
那么：
$$\mathbf{S}_b \boldsymbol{w} = c (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$$
这就意味着：**不管 $\boldsymbol{w}$ 是谁，$\mathbf{S}_b \boldsymbol{w}$ 这个向量的方向永远和 $(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$ 一模一样，只是长度差了 $c$ 倍！**

现在把它代回 (3.37) 式：
$$c (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1) = \lambda \mathbf{S}_w \boldsymbol{w}$$
两边同时左乘 $\mathbf{S}_w^{-1}$（假设 $\mathbf{S}_w$ 可逆），并把 $\lambda$ 除过去：
$$\boldsymbol{w} = \frac{c}{\lambda} \mathbf{S}_w^{-1} (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$$
还记得前面说过的吗？**我们只关心 $\boldsymbol{w}$ 的方向，不关心它的长度！** 前面的 $\frac{c}{\lambda}$ 只是一个实数倍数，乘不乘它都代表同一条直线！所以直接把这个常数扔掉（教材里等价地写成了直接令 $c = \lambda$，即式 3.38），就得到了最终极其简洁的闭式解**公式 (3.39)**：
$$\boldsymbol{w} = \mathbf{S}_w^{-1} (\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$$

*   **关于 SVD（奇异值分解）的补充**：
    在实际写代码时，有时样本特征维数 $d$ 很大、样本数 $m$ 较少，或者特征之间高度相关，会导致 $\mathbf{S}_w$ 不可逆（或者求逆时数值极不稳定）。奇异值分解是线性代数里把任意矩阵分解为三个特殊矩阵乘积的工具：$\mathbf{S}_w = U \boldsymbol{\Sigma} V^T$（其中 $U, V$ 是正交矩阵，求逆等于转置；$\boldsymbol{\Sigma}$ 是对角矩阵，求逆只需对角元素取倒数）。通过这种方式求 $\mathbf{S}_w^{-1}$（或伪逆），计算结果在计算机里更稳定、不容易报错

#### 2. 多分类 LDA 与公式 (3.43) 的证明
二分类只有 2 个类，投影到 1 条直线（1 维）就够了；如果有 $N$ 个类别（比如 3 分类：猫、狗、鸟），就要推广到多分类 LDA。
首先定义三个矩阵：
*   **全局散度矩阵 $\mathbf{S}_t$**（式 3.40）：所有样本减去全局均值 $\boldsymbol{\mu}$ 的散度。
*   **类内散度矩阵 $\mathbf{S}_w$**（式 3.41、3.42）：每个类别内部的散度矩阵之和。
*   **类间散度矩阵 $\mathbf{S}_b$**（式 3.43）：定义为 $\mathbf{S}_b = \mathbf{S}_t - \mathbf{S}_w$

教材跳过了从 (3.40)-(3.42) 推导出 (3.43) $\mathbf{S}_b = \sum_{i=1}^N m_i (\boldsymbol{\mu}_i - \boldsymbol{\mu})(\boldsymbol{\mu}_i - \boldsymbol{\mu})^T$ 的过程。我们来严格推导一次（为防止下标混淆，用 $k$ 表示第 $k$ 个类别）：

把 $\mathbf{S}_t$ 按类别分组求和，并在括号内“加一个 $\boldsymbol{\mu}_k$ 再减一个 $\boldsymbol{\mu}_k$”：
$$\begin{aligned}
\mathbf{S}_t &= \sum_{k=1}^N \sum_{\boldsymbol{x} \in X_k} (\boldsymbol{x} - \boldsymbol{\mu})(\boldsymbol{x} - \boldsymbol{\mu})^T \\
&= \sum_{k=1}^N \sum_{\boldsymbol{x} \in X_k} \big[ (\boldsymbol{x} - \boldsymbol{\mu}_k) + (\boldsymbol{\mu}_k - \boldsymbol{\mu}) \big] \big[ (\boldsymbol{x} - \boldsymbol{\mu}_k) + (\boldsymbol{\mu}_k - \boldsymbol{\mu}) \big]^T
\end{aligned}$$
像 $(a+b)^2$ 一样把括号展开成 4 项：
$$\begin{aligned}
\mathbf{S}_t &= \sum_{k=1}^N \sum_{\boldsymbol{x} \in X_k} (\boldsymbol{x} - \boldsymbol{\mu}_k)(\boldsymbol{x} - \boldsymbol{\mu}_k)^T \quad &\text{(第1项：正好等于 } \mathbf{S}_w \text{)} \\
&+ \sum_{k=1}^N \left[ \sum_{\boldsymbol{x} \in X_k} (\boldsymbol{x} - \boldsymbol{\mu}_k) \right] (\boldsymbol{\mu}_k - \boldsymbol{\mu})^T \quad &\text{(第2项：交叉项)} \\
&+ \sum_{k=1}^N (\boldsymbol{\mu}_k - \boldsymbol{\mu}) \left[ \sum_{\boldsymbol{x} \in X_k} (\boldsymbol{x} - \boldsymbol{\mu}_k)^T \right] \quad &\text{(第3项：交叉项)} \\
&+ \sum_{k=1}^N \sum_{\boldsymbol{x} \in X_k} (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T \quad &\text{(第4项)}
\end{aligned}$$
看第 2 项和第 3 项的中括号：因为 $\boldsymbol{\mu}_k$ 是第 $k$ 类 $m_k$ 个样本的平均值，所以**所有样本偏离均值之差的总和必定为 $\mathbf{0}$**（即 $\sum_{\boldsymbol{x}\in X_k} (\boldsymbol{x} - \boldsymbol{\mu}_k) = \mathbf{0}$）！所以第 2、3 项全部变成 $0$ 消失了！
再看第 4 项：$(\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$ 跟里面的 $\boldsymbol{x}$ 没关系，一共对第 $k$ 类的 $m_k$ 个样本连加 $m_k$ 次，就直接乘上 $m_k$：
$$\mathbf{S}_t = \mathbf{S}_w + \sum_{k=1}^N m_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$$
移项 $\mathbf{S}_b = \mathbf{S}_t - \mathbf{S}_w$，就得到**公式 (3.43)** 了！

---

### 第四部分：多分类矩阵求解与监督降维（对应第 4 页）

#### 1. 为什么投影矩阵是 $\mathbf{W} \in \mathbb{R}^{d \times (N-1)}$？
在 $N$ 分类问题中，我们不是只投影到 1 条直线上，而是投影到一个低维子空间中。假设原来样本有 $d$ 个特征，我们用矩阵 $\mathbf{W}$ 把它投影到 $N-1$ 维空间（$\mathbf{W}$ 的每一列代表一个投影方向，共有 $N-1$ 列）

**为什么最多只需要 $N-1$ 维？**
几何上很好理解：
*   $N=2$ 类，有 2 个类中心 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2$：两点确定一条 $1$ 维直线（$2-1=1$ 维）
*   $N=3$ 类，有 3 个类中心 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \boldsymbol{\mu}_3$：三点确定一个 $2$ 维平面（$3-1=2$ 维）
*   以此类推，$N$ 个类别中心点最多只能张成一个 $N-1$ 维的空间（在线性代数里叫做 $\mathbf{S}_b$ 的秩最多为 $N-1$）。再多出来的维度里，各类中心已经完全重合分不开了，没有区分意义

#### 2. 公式 (3.44) 中的 $\text{tr}(\cdot)$ 是什么？
$\text{tr}(A)$ 叫做矩阵的**迹（trace）**，计算极其简单——就是**方阵主对角线上所有元素的和**
当投影到 $N-1$ 维后，$\mathbf{W}^T \mathbf{S}_b \mathbf{W}$ 是一个 $(N-1) \times (N-1)$ 的矩阵：
*   它主对角线上的第 $j$ 个元素 $\boldsymbol{w}_j^T \mathbf{S}_b \boldsymbol{w}_j$，正是样本在第 $j$ 个新坐标轴上的类间散度；
*   因此，$\text{tr}(\mathbf{W}^T \mathbf{S}_b \mathbf{W})$ 的物理意义就是**所有 $N-1$ 个维度上的类间散度总和**！同理 $\text{tr}(\mathbf{W}^T \mathbf{S}_w \mathbf{W})$ 是所有维度上的类内散度总和
最大化它们的比值，就是二分类比值在多维空间的直接推广

#### 3. 广义特征值求解（公式 3.45）
同样利用拉格朗日乘子法求导，会得到公式 (3.45)：
$$\mathbf{S}_b \mathbf{W} = \lambda \mathbf{S}_w \mathbf{W}$$
如果 $\mathbf{S}_w$ 可逆，两边左乘 $\mathbf{S}_w^{-1}$，变成：
$$(\mathbf{S}_w^{-1} \mathbf{S}_b) \mathbf{W} = \lambda \mathbf{W}$$
这就是线性代数里最经典的**特征值分解**问题（回顾一下：若矩阵 $A$ 乘以向量 $\boldsymbol{v}$ 等于一个数 $\lambda$ 乘以 $\boldsymbol{v}$，即 $A\boldsymbol{v} = \lambda \boldsymbol{v}$，则 $\boldsymbol{v}$ 是特征向量，$\lambda$ 是特征值，代表在这个方向上的拉伸倍数）

在这里，特征值 $\lambda$ 越大，说明沿着这个特征向量方向投影后，“类间距离 / 类内散度”的比值就越大、分类效果越好！因此，我们只要求出矩阵 $\mathbf{S}_w^{-1} \mathbf{S}_b$ 的所有特征值，挑出**最大的 $N-1$ 个特征值对应的特征向量**，把它们一列一列拼起来，就构成了最终的投影矩阵 $\mathbf{W}$

#### 4. 最后一句话：监督降维
假设图片有 $d = 10000$ 个像素特征，一共分 $N = 10$ 类。通过 LDA 乘以投影矩阵 $\mathbf{W}^T \boldsymbol{x}$，数据一下子从 $10000$ 维被压缩到了 $N-1 = 9$ 维！而且这个压缩过程利用了样本的类别标签 $y_i$，力求让不同类别在压缩后依然好区分，所以 LDA 常被用作**有监督的降维技术**（与之相对的是第 10 章的无监督降维 PCA）

---

### 总结串联
学完这一节，你可以把 LDA 的逻辑链条压缩成 4 句话：
1. **目的**：找一条线 $\boldsymbol{w}$（或多维超平面 $\mathbf{W}$），让投影后**类间尽量远、类内尽量近**
2. **公式**：目标函数是广义瑞利商 $J = \frac{\boldsymbol{w}^T \mathbf{S}_b \boldsymbol{w}}{\boldsymbol{w}^T \mathbf{S}_w \boldsymbol{w}}$
3. **技巧**：利用放缩不变性令分母等于 $1$，再用拉格朗日乘子法求导
4. **结果**：二分类直接有公式 $\boldsymbol{w} = \mathbf{S}_w^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$；多分类取 $\mathbf{S}_w^{-1}\mathbf{S}_b$ 最大的前 $N-1$ 个特征向量


