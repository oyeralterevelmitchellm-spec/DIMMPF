# IMMPF 学习笔记

本笔记是 PF 之后的第二阶段。当前只学习 IMMPF 的入门层：多个模型/regime、模型分组、模型间交互，以及 `simulationRS.advance_one()` 的主流程。暂时不讲 DIMMPF 的可微训练、神经网络动态和 ELBO。

对应代码：

```text
Net.py
    class IMMPF

simulationRS.py
    class IMM_Particle_Filter
```

---

## PF 基础回顾

在学习 IMMPF 之前，先回顾经典粒子滤波的核心公式。

### PF 要解决的问题

系统真实状态 $x_t$ 看不见，但每一时刻能看到带噪声的观测 $y_t$。PF 的目标是估计：

$$
p(x_t \mid y_{1:t})
$$

即：已经看到从第 1 时刻到第 $t$ 时刻的所有观测后，当前状态 $x_t$ 可能在哪里。

### PF 的核心思想

用很多"候选状态"（粒子）来近似真实分布：

$$
p(x_t \mid y_{1:t})
\approx
\sum_{i=1}^{N} w_t^{(i)} \delta(x_t - x_t^{(i)})
$$

- $x_t^{(i)}$：第 $i$ 个粒子，一个"可能的状态"
- $w_t^{(i)}$：这个粒子的可信度
- $\delta(\cdot)$：可以理解成"一个点"

通俗说：PF 就是用一群带分数的小人，去代表真实状态可能在哪。

### PF 的四个核心步骤

#### 1. 状态转移（预测）

$$
x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)})
$$

每个粒子根据系统运动模型往前走一步。例如：

$$
x_t = x_{t-1} + u_t + \text{noise}
$$

#### 2. 权重更新（打分）

$$
w_t^{(i)} \propto w_{t-1}^{(i)} \cdot p(y_t \mid x_t^{(i)})
$$

粒子的新分数 = 它原来的分数 × 它解释当前观测的能力。

#### 3. 权重归一化

$$
w_t^{(i)}
=
\frac{w_t^{(i)}}{\sum_{j=1}^{N} w_t^{(j)}}
$$

把所有粒子的分数加起来变成 1。

#### 4. 状态估计

$$
\hat{x}_t = \sum_{i=1}^{N} w_t^{(i)} x_t^{(i)}
$$

每个粒子的位置乘以它的可信度，然后加起来。

### 重采样与粒子退化

时间久了以后，很多粒子的权重会变得非常小（粒子退化）。PF 会做重采样：

$$
P(x_t^{(i)*} = x_t^{(j)}) = w_t^{(j)}
$$

权重大粒子更容易被复制，权重小粒子更容易被淘汰。

有效粒子数公式：

$$
N_{\text{eff}} = \frac{1}{\sum_{i=1}^{N}(w_t^{(i)})^2}
$$

当 $N_{\text{eff}} < N/2$ 时进行重采样。

### 提议分布与重要性采样

如果不用状态转移分布 $p(x_t \mid x_{t-1})$ 采样，而是用另一个分布 $q(x_t \mid x_{t-1}, y_t)$，权重公式变为：

$$
w_t^{(i)}
\propto
w_{t-1}^{(i)}
\frac{p(y_t \mid x_t^{(i)}) \cdot p(x_t^{(i)} \mid x_{t-1}^{(i)})}
{q(x_t^{(i)} \mid x_{t-1}^{(i)}, y_t)}
$$

Bootstrap PF 是特殊情况：当 $q = p(x_t \mid x_{t-1})$ 时，分子分母的状态转移项抵消，得到经典公式：

$$
w_t^{(i)} \propto w_{t-1}^{(i)} p(y_t \mid x_t^{(i)})
$$

### 0.6/0.4 概率与模式切换的联系

你之前 Gemini 对话中的 0.6/0.4 可以理解成 PF 里的状态转移概率 $p(x_t \mid x_{t-1})$，只是状态不是连续位置，而是离散模式 $k_t$：

$$
P(k_t \mid k_{t-1}) =
\begin{cases}
0.6, & k_t = k_{t-1}+1 \\
0.4, & k_t = k_{t-1}-1
\end{cases}
$$

在 PF 中，每个粒子预测下一步时，按照这个概率规则跳。这对应本项目中 `Erlang_Switching` 类的实现。

### PF 公式总表

| 公式 | 名称 | 通俗含义 |
|---|---|---|
| $p(x_t\mid y_{1:t})$ | 后验分布 | 根据所有观测判断当前状态 |
| $p(x_t\mid x_{t-1})$ | 状态转移模型 | 系统怎么从上一时刻走到下一时刻 |
| $p(y_t\mid x_t)$ | 观测模型/似然 | 某个状态能多好地解释观测 |
| $x_t^{(i)}\sim p(x_t\mid x_{t-1}^{(i)})$ | 粒子预测 | 每个粒子往前模拟一步 |
| $\tilde{w}_t^{(i)}=w_{t-1}^{(i)}p(y_t\mid x_t^{(i)})$ | 权重更新 | 老可信度乘以当前观测匹配度 |
| $w_t^{(i)}=\frac{\tilde{w}_t^{(i)}}{\sum_j\tilde{w}_t^{(j)}}$ | 权重归一化 | 把分数变成概率 |
| $\hat{x}_t=\sum_iw_t^{(i)}x_t^{(i)}$ | 状态估计 | 用粒子做加权平均 |
| $N_{\text{eff}}=\frac{1}{\sum_i(w_t^{(i)})^2}$ | 有效粒子数 | 判断粒子是否退化 |
| $P(x_t^{(i)*}=x_t^{(j)})=w_t^{(j)}$ | 重采样 | 按权重复制好粒子、淘汰差粒子 |

PF 主线：预测粒子 → 根据观测算权重 → 归一化 → 重采样 → 估计状态。

---

### 从 PF 到 IMMPF：状态的扩展

经典 PF 只有连续状态 $x_t$。IMMPF 中，粒子状态扩展为：

$$
z_t^{(i)} = (x_t^{(i)}, r_t^{(i)})
$$

其中 $r_t^{(i)}$ 是 regime 相关状态，包含当前模式 $k_t$、历史计数或 hidden state。

预测变为两步：

$$
k_t^{(i)} \sim p(k_t \mid k_{t-1}^{(i)})
$$

$$
x_t^{(i)} \sim p(x_t \mid x_{t-1}^{(i)}, k_t^{(i)})
$$

权重更新仍然是：

$$
w_t^{(i)} \propto w_{t-1}^{(i)} p(y_t \mid x_t^{(i)}, k_t^{(i)})
$$

<font color="red">这就是 IMMPF 相对 PF 的核心扩展：先决定模式</font> $k_t$ <font color="red">，再用对应模型预测</font> $x_t$<font color="red">。</font>

### switching_dyn 的角色

`switching_dyn` 是本项目中关于离散模式 $k_t$ 的统一接口。它负责：

| 方法 | 用途 | 数学含义 |
|---|---|---|
| `init_state(...)` | 初始化模式状态 | $k_0 \sim p(k_0)$ |
| `forward(...)` / `__call__` | 采样下一模式 | $k_t \sim q(k_t \mid r_{t-1})$ |
| `get_log_probs(...)` | 计算已采样模式的 log 概率 | $\log p(k_t \mid r_{t-1})$ |
| `get_regime_probs(...)` | 输出所有候选模式概率 | $\log p(k_t=k \mid r_{t-1})$ |
| `R_0(...)` | 初始化某个指定模型的 regime 状态 | $r_0^k$ |
| `R_t(...)` | 如果当前目标模型是 $k$，更新 regime 状态 | $r_t = R(r_{t-1}, k)$ |
| `get_weight(...)` | proposal 修正权重 | $\log \frac{p}{q}$ 类修正 |

在本项目中有四种实现：

- `Markov_Switching`：固定马尔可夫切换
- `Polya_Switching`：带历史计数的切换
- `Erlang_Switching`：带持续时间/方向偏置的切换（含 0.6/0.4 非对称概率）
- `NN_Switching`：用神经网络学习切换概率（DIMMPF 的核心）

---

## IMMPF 一句话定义

```text
IMMPF = Interacting Multiple Model Particle Filter
      = 交互多模型粒子滤波
```

核心思想：

```text
PF 用一群粒子估计状态。
IMMPF 同时维护多个模型/regime，每个模型都有自己的粒子，并允许粒子在模型之间切换和交互。
```

## PF vs IMMPF

| 对比项 | PF | IMMPF |
| --- | --- | --- |
| 模型数量 | 可以先理解为一个统一模型 | 显式维护多个模型/regime |
| 粒子组织 | 所有粒子一起传播和打分 | 粒子按目标模型 `k` 分组处理 |
| 状态传播 | `M_t_proposal(x_t_1, t)` | `M_t_proposal(k, x_t_1, t)` |
| 观测似然 | `log_f_t(x_t, t)` | `log_f_t(k, x_t, t)` |
| 关键新增概念 | 无 | 模型切换概率、每个模型分配粒子、模型间交互 |

最关键的代码差异：

```text
PF:
    M_t_proposal(x_t_1, t)

IMMPF:
    M_t_proposal(k, x_t_1, t)
```

多出来的 `k` 表示：

```text
当前要用第 k 个模型来传播或评价这一组粒子。
```

## `Net.py` 中 IMMPF 的四个关键接口

### 1. `M_0_proposal(k, batches, n_samples)`

作用：

```text
为第 k 个模型生成初始粒子。
```

和 PF 的区别：

```text
PF 初始化所有粒子。
IMMPF 会按模型 k 初始化属于该模型的一组粒子。
```

理解重点：

```text
k 决定这批粒子初始属于哪个模型。
```

### 2. `M_t_proposal(k, x_t_1, t)`

作用：

```text
用第 k 个模型的动态方程，把上一时刻粒子传播到当前时刻。
```

核心公式仍然是：

```text
x_t = a[k] * x_{t-1} + b[k] + noise
```

和 PF 的区别：

```text
PF 先通过 switching_dyn 得到每个粒子自己的 new_model。
IMMPF 的滤波器外层已经决定当前目标模型 k，然后调用 M_t_proposal(k, ...)。
```

### 3. `log_f_t(k, x_t, t)`

作用：

```text
用第 k 个模型的观测方程，计算当前粒子解释观测 y_t 的能力。
```

观测模型仍然是：

```text
预测观测 = a[k] * sqrt(|x_t|) + b[k]
```

然后比较：

```text
真实观测 self.y[t]
预测观测 locs
```

二者越接近，粒子权重越高。

### 4. `get_regime_probs(x_t)`

作用：

```text
返回每个粒子从当前模型切换到各个目标模型的对数概率。
```

在 IMMPF 的交互步骤中，它回答的问题是：

```text
这个粒子被分配到目标模型 k 的概率有多大？
```

这一步不是传播状态，而是计算模型切换概率。

## `simulationRS.py` 中的 IMMPF 主流程

核心类：

```python
class IMM_Particle_Filter(nn.Module)
```

重点方法：

```python
advance_one()
```

一轮时间推进可以先理解为 7 步：

```text
1. 当前所有粒子都有权重。
2. 调用 model.get_regime_probs(self.x_t)，计算粒子切到各个模型的概率。
3. 将粒子权重加入切换概率，得到更合理的模型分配概率。
4. 对每个目标模型 k，按概率重采样一批粒子。
5. 对每个模型 k，调用 model.M_t_proposal(k, ...) 推进粒子。
6. 对每个模型 k，调用 model.log_f_t(k, ...) 根据观测打分。
7. 拼回所有模型的粒子，并归一化权重。
```

一句话：

```text
IMMPF 的 advance_one() 不是把所有粒子一次性传播，而是对每个目标模型 k 分别重采样、传播、打分，再合并。
```

## `IMM_Particle_Filter` 与 `IMMPF` 的联系

### 核心区别

| | `class IMMPF(SSM)` | `class IMM_Particle_Filter(nn.Module)` |
|---|---|---|
| **位置** | `Net.py:2303` | `simulationRS.py:35` |
| **角色** | **模型定义**（定义"规则"） | **滤波引擎**（执行"算法"） |
| **父类** | `SSM`（来自 `dpf_rs.model`） | `nn.Module` |
| **职责** | 定义状态转移、观测似然、切换策略 | 管理粒子、重采样、权重更新 |

### 怎么联系上的？

在 `main.py` 中：

```python
# 第1步：创建 IMMPF 模型（定义规则）
model = IMMPF(
    [-0.1, -0.3, ...],   # 8个模型的参数
    [0, -2, ...],
    0.1,
    Markov_Switching(...),  # 切换模型
    args.device,
)

# 第2步：把模型交给 IMM_Particle_Filter（执行滤波）
DPF = IMM_Particle_Filter(
    model,          # ← 这里！IMMPF 被传入
    2000,           # 粒子数
    Soft_Resampler_Systematic(...),
    2001,
    args.device,
    "normal",
)
```

### `IMM_Particle_Filter` 如何使用 `IMMPF`？

`IMM_Particle_Filter` 通过 `self.model` 调用 IMMPF 的方法：

| IMM_Particle_Filter 调用 | IMMPF 提供的方法 | 作用 |
|---|---|---|
| `self.model.M_0_proposal(k, ...)` | 初始粒子采样 | 时间 0 时生成粒子 |
| `self.model.M_t_proposal(k, xs[k], t)` | 状态转移采样 | 时间 t 时传播粒子 |
| `self.model.log_f_t(k, x_t, t)` | 观测似然计算 | 计算粒子权重 |
| `self.model.get_regime_probs(x_t)` | 模式概率获取 | 获取各模型的概率 |
| `self.model.log_M_t(...)` | 提议分布密度 | 权重校正 |

### 通俗比喻

> **IMMPF** = 汽车的"发动机设计图纸"（定义了怎么运转）
>
> **IMM_Particle_Filter** = "驾驶汽车"（实际执行行驶过程）
>
> 图纸（IMMPF）定义了规则，驾驶员（IMM_Particle_Filter）按照这些规则来操作。

### 继承关系图

```
dpf_rs.model.SSM          ← 来自 dpf_rs 库（状态空间模型基类）
    ↑
    └── IMMPF(SSM)        ← 定义了 8 个模型的参数、切换逻辑、观测模型
            ↑ (作为 model 参数传入)
    └── IMM_Particle_Filter(nn.Module)  ← 管理粒子、重采样、滤波流程
```

简言之：**IMMPF 是"被使用的模型"，IMM_Particle_Filter 是"使用模型的滤波器"**。

## 为什么 `IMM_Particle_Filter` 单独放在 `simulationRS.py`

### 核心结论

`simulationRS.py` **不是单独为传统 IMMPF 写的**，它是一个**通用的 IMM 粒子滤波引擎**，既服务传统 IMMPF，也服务 DIMMPF/RLPF 的超参数搜索。

### 文件职责分离

| 文件 | 职责 | 类比 |
|---|---|---|
| **Net.py** | 模型定义（状态转移、观测似然、切换策略） | 发动机设计图纸 |
| **simulationRS.py** | 滤波执行（粒子管理、重采样、时间推进） | 驾驶员 |

把"规则定义"和"算法执行"解耦的好处：
- Net.py 只关心"数学模型是什么"
- simulationRS.py 只关心"粒子怎么跑"
- 两者可以独立修改，互不干扰

### 谁在使用 `simulationRS.py`

| 使用位置 | 用途 | `IMMtype` 参数 |
|---|---|---|
| `main.py` | 传统 IMMPF 对比实验 | `"normal"` |
| `hyperparamopt.py` | RLPF 超参数优化 | `"new"` |
| `hyperparamopt.py` | DIMMPF 超参数优化 | `"new"` |

### `IMMtype` 三种模式

```python
IMMtype='normal'  # 标准 IMM 实现（传统算法，用于和 DIMMPF 对比）
IMMtype='new'     # 论文新方法（支持端到端训练，用于 RLPF/DIMMPF 超参搜索）
IMMtype='OT'      # 最优传输重采样
```

### DIMMPF 训练时为什么不用它？

DIMMPF/RLPF 的**日常训练**用的是 `dpf_rs` 库提供的 **`DPF`**（可微分粒子滤波器），而不是 `IMM_Particle_Filter`。

`IMM_Particle_Filter` 主要用于：
1. **传统 IMMPF 的基准对比**（验证 DIMMPF 的改进效果）
2. **超参数搜索阶段**（`hyperparamopt.py` 中的快速实验）

### 一句话总结

> `simulationRS.py` 是一个**通用的 IMM 粒子滤波引擎**。单独成文件是为了把"模型定义"和"滤波执行"解耦，让代码更清晰、更容易维护。

## 为什么要"交互"

如果每个模型只维护自己的粒子，不允许从其他模型吸收粒子，那么一旦当前真实 regime 切换，滤波器可能反应很慢。

IMMPF 的交互步骤允许：

```text
高权重粒子从旧模型流向新模型。
```

这样做的目的：

- 防止某些模型没有足够粒子。
- 在 regime 切换时更快响应。
- 同时利用粒子权重和模型切换概率。

## 超参数优化（Hyperparameter Optimization）

### 超参数 vs 参数

| 类型 | 定义 | 例子 | 谁来学习 |
|------|------|------|----------|
| **参数（Parameter）** | 模型内部学到的值 | 神经网络权重、状态转移系数 | 训练过程自动学习 |
| **超参数（Hyperparameter）** | 训练前人为设定的"控制开关" | 学习率、网络层数、粒子数 | 需要人工或自动调优 |

**通俗比喻**：
> 参数是汽车的"行驶状态"（速度、方向盘角度）——由驾驶员实时调整
> 
> 超参数是汽车的"设计参数"（发动机排量、轮胎尺寸）——出厂前就定好了

### 超参数优化的目的

**问题**：超参数的选择直接影响模型性能，但凭经验选往往不够好。

**解决**：用算法自动搜索最优超参数组合。

### hyperopt 如何工作

#### 核心流程

```
1. 定义搜索空间（每个超参数的取值范围）
   ↓
2. 随机或智能选一组超参数
   ↓
3. 用这组超参数训练模型，得到性能指标（如损失）
   ↓
4. 根据结果调整搜索策略（贝叶斯优化）
   ↓
5. 重复步骤 2-4，直到找到最优组合
```

#### 关键代码

```python
def optimise(function, space, max_evals):
    t = hypo.Trials()
    best = hypo.fmin(
        fn=function,           # 评估函数（如 runRLPF）
        space=space,           # 搜索空间
        algo=hypo.tpe.suggest, # TPE 算法（智能搜索）
        max_evals=max_evals,   # 最大尝试次数
        trials=t
    )
    return best
```

### 这个文件在优化什么

#### RLPF 的超参数

```python
RLPF_space = {
    'lr': ...,              # 学习率：控制训练速度
    'w_decay': ...,         # 权重衰减：防止过拟合
    'lr_gamma': ...,        # 学习率衰减因子
    'clip': ...,            # 梯度裁剪：防止梯度爆炸
    'init_scale': ...,      # 初始缩放因子
    'lamb': ...,            # 损失函数权重
    'soft_choice': ...,     # 重采样软度
    'grad_decay': ...,      # 梯度衰减
    'layers_info': ...      # 神经网络层数和隐藏层大小
}
```

#### IMMPF/DIMMPF 的超参数

```python
IMMPF_space = {
    'w_decay': ...,         # 权重衰减
    'init_scale': ...,      # 初始缩放因子
    'lambda': ...,          # 损失函数权重
    'dr': ...               # NN_Switching 的循环层维度
}
```

### 通俗比喻

> 想象你在调一杯鸡尾酒：
> 
> - **超参数** = 各种原料的比例（伏特加、果汁、糖浆）
> - **参数** = 搅拌后酒的味道（由原料比例决定）
> - **超参数优化** = 不断尝试不同比例，找到最好喝的配方
> 
> `hyperopt` 就像一个聪明的调酒师：
> - 不会盲目乱试（随机搜索）
> - 会根据上一次的味道调整下一次的配方（贝叶斯优化）
> - 最终找到最优配方（最优超参数）

### 为什么需要超参数优化

| 方法 | 优点 | 缺点 |
|------|------|------|
| **手动调参** | 快、直观 | 凭经验，可能错过最优解 |
| **网格搜索** | 全面 | 计算量大（指数级增长） |
| **随机搜索** | 比网格搜索高效 | 仍然不够智能 |
| **贝叶斯优化（hyperopt）** | 智能、高效 | 需要一定计算成本 |

### 一句话总结

> **超参数优化** = 自动搜索模型的最佳"控制开关"组合
> 
> `hyperparamopt.py` 用 `hyperopt` 库为 RLPF、DIMMPF、Transformer 自动找到最优超参数，避免手动调参的低效和盲目性。

## 第一轮自测题

1. 为什么 `IMMPF.M_t_proposal()` 比 `PF.M_t_proposal()` 多了参数 `k`？
2. `k` 表示当前粒子原来的模型，还是当前目标模型？
3. `get_regime_probs(x_t)` 返回的是状态转移结果，还是模型切换概率？
4. `simulationRS.advance_one()` 为什么要对每个模型 `k` 循环？
5. IMMPF 中"交互"指的是什么？
6. 为什么 IMMPF 比 PF 更适合 regime-switching 系统？
7. `log_f_t(k, x_t, t)` 中的 `k` 决定了什么？

---

## 自测题答案与批改

### 批改结果

| 题 | 我的答案 | 评价 |
|---|---------|------|
| 1 | 每个模型对应一个 k，IMMPF 维护多个模式的 PF | ✅ 正确 |
| 2 | 当前的目标模型 | ✅ 正确 |
| 3 | 模型切换概率 | ✅ 正确 |
| 4 | 计算模型间切换的概率 | ⚠️ 接近，但不够完整 |
| 5 | 把切换概率和粒子权重结合 | ⚠️ 这是步骤2做的事情，不完整 |
| 6 | IMMPF 同时维护多个模型 | ⚠️ 不完整 |
| 7 | 每个模型的观测似然 | ✅ 正确 |

### 正确答案详解

#### 第 1 题

**为什么 `IMMPF.M_t_proposal()` 比 `PF.M_t_proposal()` 多了参数 `k`？**

PF 只有一个统一模型，所有粒子用同一套参数传播。IMMPF 显式维护多个模型（如 8 个），每个模型有自己的动态参数 `a[k]`、`b[k]`，所以必须指明用哪个模型来传播粒子。

#### 第 2 题

**`k` 表示当前粒子原来的模型，还是当前目标模型？**

`k` 是**当前目标模型**。在 `advance_one()` 的循环中，滤波器外层已经决定了"现在要为模型 k 处理粒子"，然后调用 `M_t_proposal(k, ...)` 用模型 k 的参数传播。

#### 第 3 题

**`get_regime_probs(x_t)` 返回的是状态转移结果，还是模型切换概率？**

返回的是**模型切换概率**（对数形式）。它回答的问题是：每个粒子从当前模型切换到各个目标模型 k 的概率是多少。

#### 第 4 题

**`simulationRS.advance_one()` 为什么要对每个模型 `k` 循环？**

完整答案：IMMPF 要**把粒子按目标模型分组处理**。具体步骤：
1. 先算每个粒子切换到模型 k 的概率
2. 为模型 k 重采样一批粒子（从所有粒子中按概率挑选）
3. 用模型 k 的参数传播粒子
4. 用模型 k 的观测方程给粒子打分
5. 最后把所有模型的粒子拼回来

如果不对每个 k 循环，就无法实现"模型间交互"。

#### 第 5 题

**IMMPF 中"交互"指的是什么？**

完整答案："交互"指**高权重粒子可以从原来的模型被复制到其他模型**。

具体实现：
- 用 `p(切换概率) × w(粒子权重)` 决定每个粒子被分配到哪个模型
- 权重高的粒子对模型选择有更大发言权
- 效果：好粒子能"流向"需要它的模型，防止某些模型粒子枯竭

这就是为什么叫"交互"多模型——模型之间不是隔离的，而是通过粒子流动相互影响。

#### 第 6 题

**为什么 IMMPF 比 PF 更适合 regime-switching 系统？**

完整答案：
- PF 只用一个模型，无法显式表示"系统可能处于不同模式"
- IMMPF 显式维护多个模型，每个模型对应一种行为模式
- 更关键的是：IMMPF 允许粒子在模型间交互，当真实 regime 切换时，高权重粒子能快速流向新模型，响应更快

#### 第 7 题

**`log_f_t(k, x_t, t)` 中的 `k` 决定了什么？**

`k` 决定了**用哪个模型的观测方程计算似然**。每个模型有自己的观测参数 `a[k]`、`b[k]`，预测观测的方式不同。

---

### 下一步：进入 DIMMPF 阶段

根据学习计划，IMMPF 阶段已基本完成。接下来一周只看三处：

| 天 | 内容 | 代码位置 |
|---|---|---|
| 1 | `NN_Switching` | `Net.py:966` |
| 2 | `Simple_NN` | 找到后再看 |
| 3 | `DIMMPF.M_t_proposal` | `Net.py` 中搜索 class DIMMPF |
| 4 | `DIMMPF.log_f_t` | 同上 |
| 5 | 对比 IMMPF 与 DIMMPF | 写一张对比表 |
| 6 | `main.py` 中 DIMMPF 初始化 | `main.py:680` |
| 7 | 运行最小实验 | `python main.py --alg DIMMPF --device cpu --epochs 3 --n_runs 1 --data_dir temp` |

### 今天立刻执行的最小任务

打开 `Net.py:966` 的 `class NN_Switching`，只看：
1. `__init__` 初始化了什么
2. `forward` 输入输出是什么
3. `get_regime_probs` 返回什么

今天的目标：
> **能说清楚：NN_Switching 学的是什么，它的输入和输出分别是什么。**
