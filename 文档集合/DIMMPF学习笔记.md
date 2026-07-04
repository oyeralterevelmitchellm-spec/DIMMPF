# DIMMPF 学习笔记

本笔记是 IMMPF 之后的第三阶段。当前重点学习 DIMMPF 的核心特点：神经网络参数化、可学习噪声、端到端训练。

对应代码：

```text
Net.py
    class DIMMPF
    class NN_Switching
    class Simple_NN

main.py
    DIMMPF 初始化与训练
```

---

## DIMMPF 一句话定义

```text
DIMMPF = Differentiable Interacting Multiple Model Particle Filter
       = 可微交互多模型粒子滤波
```

核心思想：

```text
IMMPF：每个模型 k 的动态参数 a[k]、b[k] 是固定的。
DIMMPF：每个模型 k 的动态和观测模型由神经网络学习。
```

---

## IMMPF vs DIMMPF 核心对比

### 对比表格

| 对比项 | IMMPF | DIMMPF |
|--------|-------|--------|
| **状态转移** | `x_t = a[k] * x_{t-1} + b[k] + noise` | `x_t = dyn_models[k](x_{t-1}) + noise` |
| **观测模型** | `y_t = a_obs[k] * sqrt(|x_t|) + b_obs[k] + noise` | `y_t = obs_models[k](x_t) + noise` |
| **动态参数** | 固定值 `a[k]`、`b[k]` | 由神经网络 `dyn_models[k]` 学习 |
| **观测参数** | 固定值 `a_obs[k]`、`b_obs[k]` | 由神经网络 `obs_models[k]` 学习 |
| **噪声方差** | 固定值 `var_s` | 可学习的 `sd_d`（动态）、`sd_o`（观测） |
| **切换模型** | 固定的 Markov/Polya/Erlang | 可学习的 `NN_Switching` |
| **可微分** | 重采样不可微，无法端到端训练 | 使用软重采样，支持端到端训练 |

### 代码层面的区别

| 对比项 | IMMPF | DIMMPF |
|--------|-------|--------|
| **类位置** | `Net.py:2303` | `Net.py` 中搜索 `class DIMMPF` |
| **初始状态分布** | 简单的 `x_dist`（正态分布） | `init_x_dist`（可能更复杂） |
| **噪声参数** | 固定值 `var_s` | 可学习的 `sd_o`、`sd_d` |
| **观测/动态噪声** | 用一个 `var_s` | 分开：`sd_o`（观测）、`sd_d`（动态） |
| **预计算因子** | 每次重新计算 | 在初始化时预计算 `var_factor`、`pre_factor` |
| **标准化** | 没有明显的 `x_loc/x_scale` | 有 `x_loc` 和 `x_scale` 标准化 |
| **regime 状态** | 由 `switching_dyn.R_0` 初始化 | 同样由 `switching_dyn.R_0` 初始化 |

---

## DIMMPF 的三个核心变化

### 变化 1：动态模型由神经网络学习

**IMMPF**：
```python
# 固定公式
x_t = a[k] * x_{t-1} + b[k] + noise
```

**DIMMPF**：
```python
# 神经网络学习
locs = self.dyn_models[k](x_t_1[:, :, 0:1])
x_t = locs + noise
```

每个模型 k 都有一个专属的动态网络 `dyn_models[k]`，输入上一时刻状态，输出下一时刻的预测均值。

### 变化 2：观测模型由神经网络学习

**IMMPF**：
```python
# 固定公式
预测观测 = a_obs[k] * sqrt(|x_t|) + b_obs[k]
```

**DIMMPF**：
```python
# 神经网络学习
locs = self.obs_models[k](x_t[:, :, 0:1])
```

每个模型 k 都有一个专属的观测网络 `obs_models[k]`，输入当前状态，输出预测观测。

### 变化 3：切换概率由神经网络学习

**IMMPF**：
```python
# 固定的 Markov/Polya/Erlang 切换规则
P(k_t = k_{t-1}+1) = 0.6
P(k_t = k_{t-1}-1) = 0.4
```

**DIMMPF**：
```python
# NN_Switching 神经网络学习切换概率
regime_probs = self.switching_dyn.get_regime_probs(x_t)
```

`NN_Switching` 内部有循环层（hidden state），根据历史 regime 状态动态输出切换概率。

---

## DIMMPF 的噪声参数学习

### IMMPF：固定噪声

```python
self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
```

噪声方差 `var_s` 是运行前人为设定的，训练时不会改变。

### DIMMPF：可学习噪声

```python
self.var_factor = -1 / (2 * (self.sd_o**2) + 1e-6)   # 观测噪声
self.pre_factor = -(1 / 2) * pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
self.var_factor_dyn = -1 / (2 * (self.sd_d**2) + 1e-6)  # 动态噪声
self.pre_factor_dyn = -(1 / 2) * pt.log(self.sd_d**2 + 1e-6) - self.pi_fact
```

| 参数 | 含义 | 学习方式 |
|------|------|----------|
| `sd_o` | 观测噪声标准差 | 通过反向传播更新 |
| `sd_d` | 动态噪声标准差 | 通过反向传播更新 |

这些因子在初始化时预计算，用于加速后续的 `log_f_t` 和 `M_t_proposal` 计算。

---

## DIMMPF 初始化粒子时的标准化

### IMMPF：直接采样

```python
init_locs = self.x_dist.sample([batches, n_samples])
```

### DIMMPF：采样后标准化

```python
init_locs = (
    self.init_x_dist.sample([batches, n_samples])
    .to(device=self.device)
    .unsqueeze(2)
    - self.x_loc
) / self.x_scale
```

| 操作 | 含义 |
|------|------|
| `- self.x_loc` | 减去均值 |
| `/ self.x_scale` | 除以标准差 |

标准化后的数据更容易被神经网络处理（数值范围稳定）。

---

## DIMMPF 端到端训练的关键

### IMMPF：不可微

```text
重采样是离散操作（随机抽样），梯度无法传播。
```

### DIMMPF：可微

```text
使用软重采样（Soft Resampling），让梯度能够流过重采样步骤。
```

核心技巧（在 `simulationRS.py` 中）：
```python
self.log_weights = self.log_weights + weights - weights.detach()
```

前向传播：权重值不变（`weights - weights.detach() = 0`）
反向传播：梯度能流过，优化神经网络参数

---

## 通俗总结

| 概念 | IMMPF | DIMMPF |
|------|-------|--------|
| 动态模型 | 人给公式 | 神经网络学 |
| 观测模型 | 人给公式 | 神经网络学 |
| 切换概率 | 人定规则（0.6/0.4） | 神经网络学 |
| 噪声大小 | 人定固定值 | 模型自己学 |
| 能训练吗 | 不能（不可微） | 能（端到端） |

> **一句话**：DIMMPF 把 IMMPF 中所有"人定的参数/规则"都换成"神经网络学的"，然后用软重采样让整个流程可微分，实现端到端训练。

---

## 下一步学习任务

根据学习计划，接下来一周：

| 天 | 内容 | 代码位置 |
|---|---|---|
| 1 | `NN_Switching` | `Net.py:966` |
| 2 | `Simple_NN` | `Net.py` 中搜索 |
| 3 | `DIMMPF.M_t_proposal` | `Net.py` 中搜索 class DIMMPF |
| 4 | `DIMMPF.log_f_t` | 同上 |
| 5 | 对比 IMMPF 与 DIMMPF | 本笔记已完成 |
| 6 | `main.py` 中 DIMMPF 初始化 | `main.py:680` |
| 7 | 运行最小实验 | `python main.py --alg DIMMPF --device cpu --epochs 3 --n_runs 1 --data_dir temp` |

---

## DIMMPF 核心方法详解：`log_M_t`

### 方法作用

```python
def log_M_t(self, k, x_t, x_t_1, t: int):
    """提议分布的对数密度（模型k）"""
```

**一句话**：计算粒子从 $x_{t-1}$ 变到 $x_t$ 的"合理性"（对数密度）。

### 为什么需要这个？

在粒子滤波中，有一个核心问题：
> 粒子是怎么从上一时刻状态 $x_{t-1}$ 变到当前状态 $x_t$ 的？

这个变化需要一个数学公式描述，即**提议分布** $q(x_t | x_{t-1})$。

`log_M_t` 计算：
> 给定粒子从 $x_{t-1}$ 变到 $x_t$，这个变化有多"合理"？（在对数空间中）

### 代码逐行解释

```python
locs = self.dyn_models[k](x_t_1[:, :, 0:1]).squeeze()
```

| 部分 | 含义 |
|------|------|
| `dyn_models[k]` | 模型 k 的动态神经网络 |
| `x_t_1[:, :, 0:1]` | 上一时刻状态（取第一维） |
| `locs` | 预测的下一时刻位置 |

---

```python
locs = locs[:, None, :]
```

调整形状，方便后面计算。比喻：把二维纸片展开成三维，便于数据对齐。

---

```python
return (
    self.var_factor_dyn * ((x_t[:, :, None, 0] - locs) ** 2)
    + self.pre_factor_dyn
)
```

这是**高斯分布的对数密度公式**：

$$\log p(x_t | x_{t-1}) = -\frac{1}{2\sigma^2} (x_t - \mu)^2 - \frac{1}{2}\log(2\pi\sigma^2)$$

#### 公式来源推导

这个公式来自**高斯分布的概率密度函数**（概率论标准公式）。

**第一步：高斯分布的概率密度函数（PDF）**

对于均值 $\mu$、方差 $\sigma^2$ 的高斯分布：

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**第二步：取对数得到对数密度**

为什么要取对数？因为在粒子滤波中，多个概率相乘会导致数值极小（下溢），用对数把乘法变成加法更稳定。

$$\log p(x) = \log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) + \log\left(\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\right)$$

**第三步：简化公式**

利用对数性质：
- $\log(1/a) = -\log(a)$
- $\log(\exp(a)) = a$

得到：

$$\log p(x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

重新排列后就是你看到的公式：

$$\log p(x) = -\frac{1}{2\sigma^2}(x-\mu)^2 - \frac{1}{2}\log(2\pi\sigma^2)$$

| 代码部分 | 数学对应 |
|---------|---------|
| `x_t[:, :, None, 0] - locs` | $x_t - \mu$（实际位置与预测差距） |
| `** 2` | $(x_t - \mu)^2$（差距平方） |
| `var_factor_dyn` | $-1/(2\sigma^2)$（惩罚系数） |
| `pre_factor_dyn` | $-\frac{1}{2}\log(2\pi\sigma^2)$（基础分数） |

#### 为什么用这个公式？

| 原因 | 解释 |
|------|------|
| **数值稳定** | 对数空间避免乘法下溢 |
| **计算高效** | 预计算常数因子，不用每次算 |
| **梯度友好** | 对数密度是光滑函数，方便神经网络优化 |

### 通俗比喻：开车导航

> - `locs` = GPS 导航告诉你"下一个路口应该在坐标 (5, 3)"
> - `x_t` = 你实际停在了坐标 (5.1, 3.2)
> - `(x_t - locs) ** 2` = 你的实际位置和导航预测的差距平方
> - `var_factor_dyn` = 惩罚系数（差距越大，惩罚越重）
> - `pre_factor_dyn` = 基础分数（保证结果是对数概率）
>
> `log_M_t` 返回的就是："你停在这个位置有多合理？"（用对数分数表示）

### 在训练中的角色

1. **权重校正**：在 `simulationRS.py` 训练模式下：
   ```python
   self.log_weights = self.log_weights + weights - weights.detach()
   ```
   其中 `weights` 包含了 `log_M_t` 的计算结果。

2. **梯度传播**：`dyn_models[k]` 是神经网络，计算 `log_M_t` 时梯度能流过，从而优化动态模型的预测能力。

### 与 IMMPF 的对比

| 对比项 | IMMPF | DIMMPF |
|--------|-------|--------|
| `locs` 计算 | `a[k]*x_{t-1}+b[k]`（固定公式） | `dyn_models[k](x_{t-1})`（神经网络） |
| 噪声方差 | 固定值 `var_s` | 可学习的 `sd_d`（通过 `var_factor_dyn`） |
| 能优化吗 | 不能（参数固定） | 能（神经网络可训练） |

### 一句话总结

> **`log_M_t` 就是给粒子"打分"**：看它从上一状态变到当前状态，是否符合模型 k 预测的动态规律。分数越高，说明粒子"走对路了"。

---

## DIMMPF 阶段自测题

1. DIMMPF 相比 IMMPF，哪些东西变成了神经网络？
   
   **答案**：三个东西变成了神经网络：
   - **动态模型** `dyn_models[k]`：学习状态转移 $x_{t-1} \rightarrow x_t$
   - **观测模型** `obs_models[k]`：学习观测预测 $x_t \rightarrow y_t$
   - **切换模型** `NN_Switching`：学习模型切换概率 $p(k_t | r_{t-1})$

2. `dyn_models[k]` 学的是什么？
   
   **答案**：学的是**状态转移模型**。
   - 输入：上一时刻状态 $x_{t-1}$
   - 输出：下一时刻状态的预测均值 $\mu$
   - 数学含义：$x_t = \text{dyn\_models[k]}(x_{t-1}) + \text{noise}$
   - 与 IMMPF 对比：IMMPF 用固定公式 $x_t = a[k] \cdot x_{t-1} + b[k] + \text{noise}$

3. `obs_models[k]` 学的是什么？
   
   **答案**：学的是**观测模型**。
   - 输入：当前状态 $x_t$
   - 输出：预测观测的均值 $\mu_y$
   - 数学含义：$y_t = \text{obs\_models[k]}(x_t) + \text{noise}$
   - 与 IMMPF 对比：IMMPF 用固定公式 $y_t = a_{obs}[k] \cdot \sqrt{|x_t|} + b_{obs}[k] + \text{noise}$

4. `NN_Switching` 学的是什么？
   
   **答案**：学的是**模型切换概率**。
   - 输入：历史 regime 状态 $r_{t-1}$（包含模式历史、hidden state）
   - 输出：切换到各模型 $k$ 的概率 $\log p(k_t = k | r_{t-1})$
   - 与 IMMPF 对比：IMMPF 用固定的 Markov/Polya/Erlang 规则（如 0.6/0.4）
   - 核心优势：能根据历史动态调整切换策略，适应复杂场景

5. 为什么 DIMMPF 仍然需要按 k 分组处理粒子？
   
   **答案**：因为每个模型 k 有**独立的神经网络**。
   - `dyn_models[k]` 是 k 专属的动态网络
   - `obs_models[k]` 是 k 专属的观测网络
   - 不同模型的状态转移和观测预测方式不同
   - 所以粒子需要按目标模型 k 分组，分别传播和打分
   - 这与 IMMPF 的分组逻辑一致，只是公式换成了神经网络

6. `sd_o` 和 `sd_d` 分别是什么，为什么分开？
   
   **答案**：
   - `sd_d`：**动态噪声标准差**（状态转移时的不确定性）
   - `sd_o`：**观测噪声标准差**（观测时的不确定性）
   
   **为什么分开**：
   - 状态转移和观测的不确定性可能不同
   - 例如：动态很稳定（sd_d 小），但传感器噪声大（sd_o 大）
   - 分开学习让模型能分别优化两种噪声，更灵活
   - IMMPF 只用一个固定的 `var_s`，无法区分

7. DIMMPF 如何实现端到端训练（软重采样技巧）？
   
   **答案**：使用 `weights - weights.detach()` 技巧。
   
   **核心代码**（在 `simulationRS.py` 中）：
   ```python
   self.log_weights = self.log_weights + weights - weights.detach()
   ```
   
   **原理**：
   | 传播方向 | 计算结果 | 解释 |
   |---------|---------|------|
   | **前向传播** | `weights - weights = 0` | 权重值不变，相当于"加 0" |
   | **反向传播** | `梯度(weights) - 梯度(0) = 梯度(weights)` | 梯度能流过，优化神经网络 |
   
   **通俗比喻**：
   > 就像一个"单向门"：只让梯度回去，不让数值留下。
   > 前向传播时权重不变，反向传播时梯度能流过 `dyn_models[k]`、`obs_models[k]`、`NN_Switching`，实现端到端训练。
   
   **为什么需要这个**：
   - 重采样是离散操作（随机抽样），本身不可微分
   - 软重采样技巧让梯度能"绕过"重采样步骤，继续传播
   - 这是 DIMMPF 能训练的核心关键技术