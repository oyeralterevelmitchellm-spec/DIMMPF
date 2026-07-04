# PF 基础学习笔记

本笔记只解决第一阶段问题：先把 `PF` 的基础粒子滤波数据流看懂。暂时不讲 `k_t / r_t` 的完整论文含义，不讲可微训练，也不讲 ELBO。后面学习 `IMMPF` 和 `DIMMPF` 时，会在这个基础上继续加“多个模型”和“切换机制”。

对应代码位置：`Net.py` 中的 `class PF(SSM)`。

## 先记住的最小循环

粒子滤波可以先理解为一句话：

```text
维护很多个可能状态，用观测 y_t 给这些可能状态打分，保留更合理的状态。
```

在 `PF` 里，最小循环是：

```text
M_0_proposal()
    -> 生成 t=0 的初始粒子

x_{t-1}
    -> M_t_proposal()
    -> x_t
    -> log_f_t()
    -> 当前粒子的观测得分/权重
```

也可以画成：

```text
上一时刻粒子 x_{t-1}
        |
        v
M_t_proposal(): 按状态转移方程生成当前粒子 x_t
        |
        v
log_f_t(): 用当前观测 y_t 给 x_t 打分
        |
        v
权重归一化和重采样由粒子滤波器框架处理
```

当前阶段只需要抓住三件事：

- `M_0_proposal()`：初始粒子从哪里来。
- `M_t_proposal()`：粒子如何从 `t-1` 走到 `t`。
- `log_f_t()`：粒子如何根据观测 `y_t` 得分。

## 论文变量到代码变量

| 概念 | 直觉 | `PF` 代码里的对应 |
| --- | --- | --- |
| `x_t` | 隐藏状态，真实但不能直接看到 | 粒子张量里的位置部分，常见为 `x_t[:, :, 0]` |
| `y_t` | 当前观测，带噪声但能看到 | `self.y[t]` |
| 粒子 | 对隐藏状态的一种猜测 | `x_t` 张量中的一个样本 |
| 观测似然 | 某个粒子解释当前观测的能力 | `log_f_t()` 返回的对数分数 |
| 权重 | 粒子的可信程度 | 由滤波器框架根据 `log_f_t()` 等结果维护 |

这里的 `x_t` 不等于一个单独数字，而是一批粒子：

```text
x_t.shape = (batch_size, n_particles, state_dim)
```

在 `PF` 中，粒子至少包含：

```text
[位置, 模型/模式信息]
```

所以代码中会看到：

```python
x_t[:, :, 0]   # 粒子的连续状态/位置
x_t[:, :, 1:]  # 粒子携带的模型或模式信息
```

## 为什么写 `x_t_1[:, :, 0]`

`x_t_1` 是上一时刻的粒子张量，通常是三维：

```text
x_t_1.shape = (B, N, D)
```

含义：

```text
B = batch_size，同时处理多少条序列
N = n_particles，每条序列有多少个粒子
D = state_dim，每个粒子内部存了多少个信息
```

在 `PF` 这里，一个粒子可以先理解成：

```text
x_t_1[b, n, :] = [position, model_info]
```

也就是：

```text
第 b 条序列的第 n 个粒子 = [位置, 模型/模式信息]
```

所以：

```python
x_t_1[:, :, 0]
```

意思是：

```text
第一个 :  取所有 batch
第二个 :  取所有粒子
第三个 0  取每个粒子的第 0 个分量，也就是位置
```

结果形状是：

```text
(B, N)
```

举例：

```text
x_t_1.shape = (2, 3, 2)

x_t_1 =
[
  [
    [1.0, 0],
    [1.5, 2],
    [0.7, 1]
  ],
  [
    [-0.2, 3],
    [0.4, 1],
    [1.1, 0]
  ]
]
```

则：

```python
x_t_1[:, :, 0]
```

得到：

```text
[
  [1.0, 1.5, 0.7],
  [-0.2, 0.4, 1.1]
]
```

它取出了所有粒子的“位置”，形状是：

```text
(2, 3)
```

而：

```python
x_t_1[:, :, 1:]
```

得到：

```text
[
  [[0], [2], [1]],
  [[3], [1], [0]]
]
```

它取出了所有粒子的“模型/模式信息”，形状是：

```text
(2, 3, 1)
```

### `0` 和 `0:1` 的区别

```python
x_t_1[:, :, 0]
```

会降维：

```text
(B, N, D) -> (B, N)
```

而：

```python
x_t_1[:, :, 0:1]
```

会保留第三维：

```text
(B, N, D) -> (B, N, 1)
```

在 `M_t_proposal()` 中使用：

```python
x_t_1[:, :, 0]
```

是为了先拿到 `(B, N)` 形状的位置，方便和 `scaling`、`bias`、`noise` 做逐粒子计算。计算完成后再用：

```python
.unsqueeze(2)
```

把结果变回：

```text
(B, N, 1)
```

这样才能和 `new_models` 在最后一维拼接：

```python
pt.cat((new_pos, new_models), dim=2)
```

记忆版总结：

```text
x_t_1.shape = (B, N, D)

x_t_1[:, :, 0]
    = 取所有序列、所有粒子的第 0 个分量
    = 上一时刻位置 x_{t-1}
    = 形状 (B, N)

x_t_1[:, :, 1:]
    = 取所有序列、所有粒子的模型/模式信息
    = 形状 (B, N, D-1)

先用 x_t_1[:, :, 0] 做状态转移计算，
再用 unsqueeze(2) 恢复成 (B, N, 1)，
最后和 new_models 拼成新的粒子。
```

## PF 三个核心方法

### 1. `M_0_proposal()`: 生成初始粒子

作用：

```text
在 t=0 时刻，先生成很多个可能的初始状态。
```

关键代码：

```python
init_locs = (
    self.init_x_dist.sample([batches, n_samples])
    .to(device=self.device)
    .unsqueeze(2)
)
```

含义：

- 从 `Uniform(-0.5, 0.5)` 采样初始位置。
- 每个粒子得到一个初始 `x_0`。
- `unsqueeze(2)` 是为了把形状整理成 `(batches, n_samples, 1)`，方便后面拼接。

然后：

```python
init_regimes = self.switching_dyn.init_state(batches, n_samples)
return pt.cat((init_locs, init_regimes), dim=2)
```

含义：

- `init_regimes` 是每个粒子的初始模型/模式信息。
- `cat` 把位置和模式信息拼成完整粒子。

阶段性理解：

```text
M_0_proposal() = 生成初始位置 + 生成初始模式 + 拼成初始粒子
```

### 2. `M_t_proposal()`: 从上一时刻传播到当前时刻

作用：

```text
根据状态转移模型，把 x_{t-1} 推进到 x_t。
```

关键代码：

```python
new_models = self.switching_dyn(x_t_1[:, :, 1:], t)
index = new_models[:, :, 0].to(int)
```

含义：

- `x_t_1[:, :, 1:]` 取出上一时刻粒子的模式信息。
- `switching_dyn(...)` 根据切换模型得到当前时刻的新模型编号。
- `index` 是当前粒子属于哪个模型。

然后：

```python
scaling = self.a[index]
bias = self.b[index]
new_pos = (scaling * x_t_1[:, :, 0] + bias).unsqueeze(2) + noise
```

含义：

- 每个模型都有自己的 `a[k]` 和 `b[k]`。
- 当前粒子属于哪个模型，就用哪个模型的状态转移方程。
- 状态转移形式是：

```text
x_t = a[k] * x_{t-1} + b[k] + noise
```

最后：

```python
return pt.cat((new_pos, new_models), dim=2)
```

含义：

- 把新的位置和新的模型信息拼回完整粒子。

阶段性理解：

```text
M_t_proposal() = 选当前模型 k + 用模型 k 的动态方程生成新位置 + 拼成新粒子
```

## 论文中的 `a`、`b` 是什么

论文 Section 5.1 中给了 8 个 regime/model，每个 regime 都有一组参数：

```text
a = [-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9]
b = [0, -2, 2, -4, 0, 2, -2, 4]
```

含义：

```text
模型 1 使用 a1=-0.1, b1=0
模型 2 使用 a2=-0.3, b2=-2
模型 3 使用 a3=-0.5, b3=2
...
模型 8 使用 a8=0.9, b8=4
```

论文从 1 开始编号，代码中 Python 从 0 开始编号。

### 在状态转移中

论文状态转移公式：

```text
x_t ~ Normal(a[k_t] * x_{t-1} + b[k_t], sigma^2)
```

代码对应 `PF.M_t_proposal()`：

```python
scaling = self.a[index]
bias = self.b[index]
new_pos = (scaling * x_t_1[:, :, 0] + bias).unsqueeze(2) + noise
```

对应关系：

```text
scaling = a[k_t]
bias    = b[k_t]

new_pos = a[k_t] * x_{t-1} + b[k_t] + noise
```

所以在状态转移里：

```text
a 控制上一时刻状态对当前状态的影响强度和方向
b 控制当前状态的整体偏移
```

### 在观测模型中

论文观测模型公式：

```text
y_t ~ Normal(a[k_t] * sqrt(|x_t|) + b[k_t], sigma^2)
```

代码对应 `PF.log_f_t()`：

```python
scaling = self.a[index]
bias = self.b[index]
locs = scaling * pt.sqrt(pt.abs(x_t[:, :, 0]) + 1e-7) + bias
```

对应关系：

```text
locs = a[k_t] * sqrt(|x_t|) + b[k_t]
```

这里 `locs` 是粒子预测出来的观测均值。`log_f_t()` 会比较：

```text
真实观测 self.y[t]
预测观测 locs
```

二者越接近，粒子权重越高；二者差得越远，粒子权重越低。

### 直观理解

`a` 可以理解为斜率或缩放：

```text
a > 0：状态影响方向为正
a < 0：状态影响方向反转
|a| 越大：上一状态对当前状态影响越强
```

`b` 可以理解为偏置：

```text
b > 0：整体往上偏移
b < 0：整体往下偏移
b = 0：没有额外偏移
```

例如某个 regime 的参数是：

```text
a = -0.9
b = -4
```

那么状态转移均值大致是：

```text
x_t ≈ -0.9 * x_{t-1} - 4
```

表示当前状态会强烈受上一状态影响，但方向反转，并整体向负方向偏移。

记忆版总结：

```text
a、b 是每个 regime/model 自己的参数。

当前粒子属于哪个模型 k，
代码就取 a[k]、b[k]。

M_t_proposal() 用 a[k]、b[k] 传播状态。
log_f_t() 用 a[k]、b[k] 预测观测并计算似然。
```

### 3. `log_f_t()`: 用观测给粒子打分

作用：

```text
判断当前粒子 x_t 是否能解释当前观测 y_t。
```

关键代码：

```python
index = x_t[:, :, 1].to(int)
scaling = self.a[index]
bias = self.b[index]
locs = scaling * pt.sqrt(pt.abs(x_t[:, :, 0]) + 1e-7) + bias
```

含义：

- 先取当前粒子的模型编号。
- 根据模型编号选对应的 `a[k]` 和 `b[k]`。
- 用观测方程预测“如果这个粒子是真的，应该看到什么观测”。

观测方程是：

```text
预测观测 = a[k] * sqrt(|x_t|) + b[k]
```

然后：

```python
return self.var_factor * ((self.y[t] - locs) ** 2)
```

含义：

- `self.y[t]` 是真实看到的当前观测。
- `locs` 是粒子预测出来的观测。
- 二者越接近，误差越小，log likelihood 越高。
- 二者差得越远，这个粒子的权重就应该越低。

阶段性理解：

```text
log_f_t() = 用观测 y_t 检查粒子 x_t 是否靠谱
```

## Radon-Nikodym 导数在这里是什么

Radon-Nikodym 导数在粒子滤波里可以先理解成：

```text
重要性采样里的纠偏系数
```

原因是：理论上我们希望粒子来自目标分布，但代码里实际可能从另一个更方便的提议分布采样。两者不一致时，粒子权重必须补一个比例：

<font color="red">`纠偏系数 = 目标分布 / 提议分布`</font>

这就是 Radon-Nikodym 导数在本项目里的实际作用。

用粒子滤波符号写：

<font color="red">`R_t = p(x_t | x_{t-1}) / q(x_t | x_{t-1}, y_t)`</font>

其中：

- `p(...)`：真实动态/目标分布。
- `q(...)`：代码实际用来生成粒子的提议分布。
- `R_t`：两者的修正比例。

代码中通常使用对数形式：

```text
log_R_t = log p(...) - log q(...)
```

这样数值更稳定，也方便和其他 log 权重相加。

### 和 `log_f_t()` 的关系

`log_f_t()` 负责观测打分：

```text
log_f_t = log p(y_t | x_t)
```

`log_R_t()` 负责提议分布纠偏：

```text
log_R_t = log(目标分布 / 提议分布)
```

所以在通用权重更新里，二者会一起出现：

```text
log_weight_t = log_weight_{t-1} + log_R_t + log_f_t
```

对应 `dpf_rs/model.py`：

```python
def log_G_t_guided(self, x_t, x_t_1, t: int):
    return self.log_R_t(x_t, x_t_1, t) + self.log_f_t(x_t, t)
```

也就是：

```text
完整权重 = Radon-Nikodym 修正项 + 观测似然
```

### Bootstrap 情况为什么经常不需要纠偏

如果使用 Bootstrap 粒子滤波：

```text
提议分布 = 真实状态转移分布
```

也就是：

```text
q(x_t | x_{t-1}) = p(x_t | x_{t-1})
```

那么：

```text
R_t = p / q = 1
log_R_t = log(1) = 0
```

所以 Bootstrap 下权重主要由 `log_f_t()` 决定。

在 `dpf_rs/simulation.py` 中可以看到类似逻辑：

```python
if self.model.alg == self.model.PF_Type.Bootstrap:
    self.log_weights += self.model.log_f_t(self.x_t, self.t)
elif self.model.alg == self.model.PF_Type.Guided:
    self.log_weights += self.model.log_G_t_guided(self.x_t, self.x_t_1, self.t)
```

意思是：

```text
Bootstrap:
    只加 log_f_t()

Guided:
    加 log_R_t() + log_f_t()
```

### 在 `Net.py` 的 PF 中

`PF.log_R_0()` 返回 0：

```python
return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)
```

含义：

```text
log_R_0 = 0
R_0 = 1
```

也就是初始时刻不额外纠偏。

`PF.log_R_t()` 中：

```python
return self.switching_dyn.get_log_probs(x_t[:, :, 1:], x_t_1[:, :, 1:])
```

它取出当前和上一时刻粒子的模型/模式信息：

```python
x_t[:, :, 1:]
x_t_1[:, :, 1:]
```

然后计算：

```text
log P(当前模型 | 上一时刻模型)
```

也就是模型切换这部分的对数概率。

记忆版总结：

```text
M_t_proposal()
    负责生成粒子

log_f_t()
    负责用观测 y_t 给粒子打分

log_R_t()
    负责修正“实际怎么生成粒子”和“理论目标分布”之间的差异
```

## 三个方法的关系

把三个方法串起来：

```text
M_0_proposal()
    生成初始粒子 x_0

M_t_proposal(x_{t-1}, t)
    根据模型动态生成 x_t

log_f_t(x_t, t)
    根据观测 y_t 给 x_t 打分
```

粒子滤波器框架会在这些分数基础上做权重归一化和重采样。`PF` 类本身主要负责定义“怎么生成粒子”和“怎么给粒子打分”。

## 和后续 IMMPF / DIMMPF 的连接点

先不要展开后续细节，只记住这个关系：

```text
PF:
    一个基础粒子滤波模型，重点是状态传播和观测打分。

IMMPF:
    在 PF 基础上显式处理多个 regime/model，并让不同模型之间交互。

DIMMPF:
    在 IMMPF 基础上，把切换动态、状态转移或观测模型做成可学习组件，并参与端到端训练。
```

后续学习 `IMMPF` 和 `DIMMPF` 时，仍然离不开这三个动作：

```text
初始化粒子 -> 传播粒子 -> 根据观测打分
```

只是后续会加入：

- 多个模型 `k`。
- regime 切换概率。
- 每个模型分配粒子。
- 可微分训练。

## 附录：提议分布 vs 先验分布

在粒子滤波中，**提议分布（Proposal Distribution）** 和 **先验分布（Prior Distribution）** 是两个密切相关但不同的概念。

### 核心区别

| 特性 | 先验分布 $p(x_t \| x_{t-1})$ | 提议分布 $q(x_t \| x_{t-1}, y_t)$ |
|------|------------------------------|-----------------------------------|
| **定义** | 状态转移的概率模型 | 用于采样的近似分布 |
| **是否使用观测** | ❌ 不使用当前观测 $y_t$ | ✅ 可以使用当前观测 $y_t$ |
| **目的** | 描述系统动态 | 生成更好的粒子位置 |
| **数学形式** | 由系统模型决定 | 可以灵活设计 |

### 两种粒子滤波类型

#### 1. Bootstrap 粒子滤波（标准形式）

```
提议分布 = 先验分布
q(x_t | x_{t-1}, y_t) = p(x_t | x_{t-1})
```

**特点**：
- 简单直接，只根据系统动态传播粒子
- 不使用观测信息引导粒子
- 权重更新：$w_t \propto w_{t-1} \cdot p(y_t \| x_t)$

**缺点**：当似然分布与先验分布重叠较小时，粒子容易"跑偏"

#### 2. Guided 粒子滤波（优化形式）

```
提议分布 ≠ 先验分布
q(x_t | x_{t-1}, y_t) ∝ p(y_t | x_t) · p(x_t | x_{t-1})
```

**特点**：
- 利用观测信息 $y_t$ 引导粒子向高似然区域移动
- 粒子质量更高，需要更少的粒子
- 需要计算 R-N 导数进行权重修正

**权重修正**：
$$R_t = \frac{p(x_t \| x_{t-1})}{q(x_t \| x_{t-1}, y_t)}$$

### 在代码中的体现

在 `Net.py` 中：

```python
if dyn == "Boot":
    self.alg = self.PF_Type.Bootstrap  # 提议 = 先验
else:
    self.alg = self.PF_Type.Guided     # 提议 ≠ 先验
```

**`log_R_t` 的作用**：
- **Bootstrap**：$R_t = 1$（不需要修正）
- **Guided**：$R_t = \frac{p(x_t \| x_{t-1})}{q(x_t \| x_{t-1}, y_t)}$（需要修正）

### 总结

| 问题 | 答案 |
|------|------|
| 提议分布 **可以** 是先验分布 | ✅ 是的（Bootstrap 情况） |
| 提议分布 **必须** 是先验分布 | ❌ 不是（Guided 情况更优） |
| 提议分布 **总是** 先验分布 | ❌ 错误 |

**关键理解**：提议分布是我们**主动设计**的采样分布，而先验分布是系统**固有的**转移概率。在 Bootstrap 滤波中二者相同，但在 Guided 滤波中，提议分布会利用观测信息，比先验分布更"聪明"。

---

## 学习检查结果与易错点

本节记录 PF 第一轮学习检查的结论。当前结论：PF 可以进入下一阶段，但以下点需要长期记住。

### 已掌握

- 粒子表示隐藏状态的一种可能性。
- `x_t` 是状态，`y_t` 是观测。
- `x_t_1[:, :, 0]` 取位置；当 `x_t_1.shape = (4, 200, 2)` 时，结果形状是 `(4, 200)`。
- `unsqueeze(2)` 用于恢复第三维，方便后续 `pt.cat(..., dim=2)` 拼接。
- `M_0_proposal()` 生成初始粒子和初始模型/模式信息。
- `M_t_proposal()` 输入上一时刻粒子，输出当前时刻新粒子。
- `a`、`b` 是不同 regime/model 对应的参数。
- `index` 用于根据当前模型编号查找对应的 `a[index]` 和 `b[index]`。
- `switching_dyn` 根据切换机制生成新的模型/模式信息。
- `log_R_t()` 用于补偿提议采样和目标动态分布之间的差异。

### 需要修正的点

1. 为什么要维护很多粒子？

```text
真实状态不可见，而且存在不确定性。
一个粒子只代表一种可能状态。
很多粒子一起近似整个后验分布 p(x_t | y_0:t)。
```

不要把粒子滤波理解成只找一个最优状态。更准确是：

```text
用一群可能状态表示当前不确定性。
```

2. `x_t_1[:, :, 1:]` 的形状

如果：

```text
x_t_1.shape = (4, 200, 2)
```

那么：

```python
x_t_1[:, :, 1:]
```

形状是：

```text
(4, 200, 1)
```

含义是：

```text
取每个粒子除位置以外的模型/模式信息。
```

3. Bootstrap PF 的提议分布不是后验

正确说法：

```text
Bootstrap PF 的提议分布等于状态转移先验 p(x_t | x_{t-1})。
```

不是：

```text
提议分布等于后验分布。
```

后验还需要观测似然：

```text
p(x_t | y_0:t) ∝ p(y_t | x_t) * p(x_t | y_0:t-1)
```

所以 Bootstrap 下通常不需要额外的 `log_R_t()` 修正，但仍然必须使用 `log_f_t()` 给粒子打分。

### 进入 IMMPF 前必须掌握的 4 句话

```text
1. 粒子滤波用很多粒子近似后验分布，而不是只维护一个状态估计。
2. M_t_proposal() 负责生成/传播粒子。
3. log_f_t() 负责用观测 y_t 给粒子打分。
4. log_R_t() 负责修正提议分布和目标动态分布之间的差异。
```

## 自测题

进入 `IMMPF` 前，先确认你能回答这些问题：

1. `M_0_proposal()` 生成的是什么？为什么要生成很多个？
2. `M_t_proposal()` 的输入和输出分别是什么？
3. 在 `M_t_proposal()` 里，`a[index]` 和 `b[index]` 的作用是什么？
4. `log_f_t()` 为什么必须用到 `self.y[t]`？
5. 如果某个粒子预测出来的观测 `locs` 和真实观测 `self.y[t]` 差很远，这个粒子的权重应该变高还是变低？
6. `M_t_proposal()` 负责更新观测 `y_t` 吗？如果不是，它更新什么？
7. `log_f_t()` 是生成新粒子，还是给已有粒子打分？
8. 为什么说 `PF` 是后续学习 `IMMPF/DIMMPF` 的基础？
9. **Bootstrap 滤波和 Guided 滤波的主要区别是什么？**
10. **`log_R_t()` 在什么情况下等于 1（或 log 值为 0）？**
