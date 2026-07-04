# 学习DIMMPF的步骤

**User:** Anonymous  
**Created:** 6/26/2026 17:10:18  
**Updated:** 6/26/2026 17:18:44  
**Exported:** 6/26/2026 17:24:56  
**Link:** [https://chatgpt.com/c/6a3e41f8-5ca0-83ea-82d2-93eff493c27a](https://chatgpt.com/c/6a3e41f8-5ca0-83ea-82d2-93eff493c27a)  

## Response:
6/26/2026, 5:16:19 PM

我会把你的现有笔记、GitHub 仓库结构、论文/代码学习路线合并成一份“从恢复状态到可改算法”的计划。重点不是重新从零学，而是先定位你卡在哪里，再安排最小可执行步骤。

我已经确认仓库是公开的，默认分支是 `main`，README 说明它是论文 *Differentiable Interacting Multiple Model Particle Filtering* 的实验代码。仓库入口主要是 `main.py`、`Net.py`、`SimulationRS.py/simulationRS.py`、`trainingRS.py` 和 `dpf-rs/`。

> **获取 dpf_rs 模型和相关结果**
>
> [Differentiable Interacting Multiple Model Particle Filtering](https://arxiv.org/abs/2410.00620)
> [Particle filter](https://en.wikipedia.org/wiki/Particle_filter)
>
> **搜索开放的arxiv资源**
>
> [Differentiable Interacting Multiple Model Particle Filtering](https://arxiv.org/abs/2410.00620)
>
> **审查文件和引用的格式问题**
>
> 在使用某些工具时，代码文件的引用可能没有显示行号，可能需要使用 GitHub 文件引用并加上行号。GitHub 支持带行号的引用，需注意在不同操作系统上的大小写差异，例如在 Linux 上区分大小写。注意 README 中的大小写不一致问题。
>
> **制定计划及下一步**
>
> 似乎不需要提取 dpf_rs 目录，搜索 "requirements" 也没结果。虽然他们请求帮助制定计划，但不必创建文档，除非有要求。可以先形成一个可复用的计划，并提供必要的引用，像 arxiv 摘要和代码。接下来需注意 “恢复期 3天，主线4周”。
>
> 已思考 1m 19s

## 先给结论

你**不要重新从 PF 从头学一遍**，也不要现在直接冲 DIMMPF 的训练和 ELBO。你现在最合理的下一步是：

> **先用 3-5 天把 IMMPF 的 `advance_one()` 主流程彻底打通，然后再进入 DIMMPF。**

从你上传的进度看，你的 PF 第一轮已经完成，当前阶段本来就是 IMMPF，目标是理解“PF 如何扩展成多个 regime/model 的粒子滤波”，暂时不进入 DIMMPF 的可微训练和 ELBO。`学习总路线与进度.md`  
你之前卡住的主要原因，大概率不是你不适合学，而是**同时看论文公式、DIMMPF、训练、ELBO、神经网络、代码细节，层次混在一起了**。

---

# 1. 你当前到底学到哪里了？

你的路线文件里原本定的是：

```text
PF -> IMMPF -> DIMMPF -> 训练与损失 -> 论文公式到代码映射
```

其中 PF 已完成第一轮，IMMPF 是当前阶段，DIMMPF、训练与损失、论文映射还没进入。`学习总路线与进度.md`

所以你现在不是“完全忘了”，而是处在这个位置：

```text
PF 基础：已有基础，但需要快速热身
IMMPF：正在学，应该作为下一步主攻
DIMMPF：先不要急
训练/ELBO：暂时放后面
改进算法写小论文：最后再开始
```

你已经掌握的 PF 核心应当保留为三句话：

```text
M_0_proposal()：生成初始粒子
M_t_proposal()：传播粒子
log_f_t()：根据观测 y_t 给粒子打分
```

这些也是 IMMPF 和 DIMMPF 的共同骨架。`PF基础学习笔记.md`

---

# 2. 这篇论文和代码的核心关系

论文 *Differentiable Interacting Multiple Model Particle Filtering* 提出的目标是：针对存在随机、突发 regime 切换的系统，用可微粒子滤波做参数学习，并同时学习各个行为模式和控制跳变的模型。([arXiv](https://arxiv.org/abs/2410.00620))

仓库 README 明确说，这个仓库是用于生成该论文实验结果的代码；入口是 `main.py`，模型实现主要在 `Net.py`，滤波算法在 `simulationRS.py`，训练测试在 `trainingRS.py`。

代码中最重要的对应关系是：

| 层次 | 论文概念 | 代码位置 | 你要看懂什么 |
|---|---|---|---|
| PF | 普通粒子滤波 | `Net.py -> class PF` | 粒子如何传播、打分 |
| IMMPF | 多模型交互粒子滤波 | `Net.py -> class IMMPF` + `simulationRS.py` | 为什么每个模型 `k` 分组重采样 |
| DIMMPF | 可微 + 神经网络版本 IMMPF | `Net.py -> class DIMMPF` | 动态模型和观测模型如何由神经网络学习 |
| 训练 | MSE + likelihood/ELBO | `trainingRS.py -> e2e_train()` | 损失怎么组合，参数怎么更新 |

---

# 3. 你下一步该做什么？

## 第一件事：不要先看整篇论文

你现在应该先做这个：

```text
读代码主流程 -> 再回论文找对应公式
```

而不是：

```text
先读论文公式 -> 再硬找代码
```

因为这篇文章的难点不是某一个公式，而是这些东西同时出现：

```text
PF
IMM
regime switching
resampling
可微分
神经网络参数化
ELBO / likelihood loss
```

如果直接看论文，会很容易卡在符号上。

---

# 4. 具体学习计划

## 阶段 0：恢复记忆，1-2 天

目标：把 PF 的最小循环重新捡起来，不扩展。

只复习这些：

```text
1. x_t 是粒子状态，不是单个数
2. y_t 是观测
3. M_t_proposal() 负责传播
4. log_f_t() 负责打分
5. 权重归一化和重采样由滤波器框架处理
```

重点看你自己的 `PF基础学习笔记.md`，不要新开资料。你之前笔记里已经明确记录了：PF 是后续 IMMPF/DIMMPF 的基础，后续仍然离不开“初始化粒子 -> 传播粒子 -> 根据观测打分”这条主线。`PF基础学习笔记.md`

验收标准：

```text
你能不看笔记说清：
M_0_proposal、M_t_proposal、log_f_t 分别干什么。
```

如果说不清，再复习；如果能说清，立刻进入 IMMPF。

---

## 阶段 1：IMMPF，3-5 天

这是你现在真正该做的阶段。

### 你要先理解一句话

```text
IMMPF = PF + 多个模型 k + 模型之间交互
```

你的 IMMPF 笔记里已经写得很关键：PF 是所有粒子一起传播，而 IMMPF 是粒子按目标模型 `k` 分组处理；最关键的代码差异是：

```text
PF:
    M_t_proposal(x_t_1, t)

IMMPF:
    M_t_proposal(k, x_t_1, t)
```

其中 `k` 表示“当前要用第 k 个模型来传播或评价这一组粒子”。`IMMPF学习笔记.md`

### 每天任务

| 天数 | 任务 | 只看这些 |
|---|---|---|
| Day 1 | 看 `Net.py -> class IMMPF` | `M_0_proposal(k,...)`, `M_t_proposal(k,...)`, `log_f_t(k,...)`, `get_regime_probs()` |
| Day 2 | 看 `simulationRS.py -> initialise()` | 每个模型如何初始化粒子 |
| Day 3 | 看 `simulationRS.py -> advance_one()` | 为什么对每个 `k` 重采样、传播、打分 |
| Day 4 | 手画流程图 | 从 `regime_probs` 到 `log_weights` |
| Day 5 | 自测 | 不看代码解释一轮 IMMPF |

代码中 `IMMPF.M_0_proposal(k,...)` 会为指定模型初始化粒子和 regime 状态；`M_t_proposal(k,...)` 使用第 `k` 个模型的 `a[k]`、`b[k]` 传播状态；`log_f_t(k,...)` 用第 `k` 个模型的观测方程算似然；`get_regime_probs()` 返回模型切换概率。

`simulationRS.py` 里的 `advance_one()` 是主流程：先算 `regime_probs`，再把粒子权重加进去，得到各目标模型的重采样权重，然后对每个模型 `k` 重采样、调用 `M_t_proposal(k,...)`，最后用 `log_f_t(k,...)` 更新权重并归一化。

### IMMPF 阶段验收题

你必须能回答：

```text
1. 为什么 IMMPF 的 M_t_proposal 多了参数 k？
2. get_regime_probs() 返回的是状态 x_t，还是模型切换概率？
3. advance_one() 为什么要 for k in range(Nk)？
4. 什么叫“交互”？
5. 高权重粒子为什么可以从旧模型流向新模型？
```

这些答不清，就不要进入 DIMMPF。

---

## 阶段 2：DIMMPF，5-7 天

IMMPF 明白后，再看 DIMMPF。

### 先抓住核心区别

```text
IMMPF:
    每个模型 k 的动态是固定公式：
    x_t = a[k] * x_{t-1} + b[k] + noise

DIMMPF:
    每个模型 k 的动态由神经网络学习：
    x_t = dyn_models[k](x_{t-1}) + noise
```

代码中 `DIMMPF` 定义了 `dyn_models` 和 `obs_models` 两组 `ModuleList`，每个 regime/model 都有一个动态网络和一个观测网络。

它的核心方法是：

```python
M_t_proposal(k, x_t_1, t)
```

里面用：

```python
locs = self.dyn_models[k](x_t_1[:, :, 0:1])
```

生成状态转移均值，再加噪声；`log_f_t(k, x_t, t)` 则用：

```python
locs = self.obs_models[k](x_t[:, :, 0:1])
```

预测观测并计算似然。

### DIMMPF 阶段你只看三处

```text
Net.py:
    NN_Switching
    Simple_NN
    DIMMPF

simulationRS.py:
    advance_one()

main.py:
    DIMMPF 初始化部分
```

不要一开始就看 `DIMMPF_redefined`。那是后面理解 likelihood/ELBO 时用的。

### DIMMPF 阶段验收题

```text
1. DIMMPF 相比 IMMPF，哪些东西变成了神经网络？
2. dyn_models[k] 学的是什么？
3. obs_models[k] 学的是什么？
4. NN_Switching 学的是什么？
5. 为什么 DIMMPF 仍然需要按 k 分组？
```

---

## 阶段 3：训练与实验，1 周

这个阶段才开始跑代码。

README 说明 `main.py` 支持这些算法：

```text
DIMMPF
DIMMPF-OT
DIMMPF-N
RLPF
Transformer
LSTM
IMMPF
```

并且参数里包括学习率、权重衰减、梯度裁剪、`lambda`、网络层数、隐藏层大小、epoch、运行次数和数据目录等。

`main.py` 中实际也是这样设置的：`--alg` 可选 `RLPF/LSTM/Transformer/DIMMPF/DIMMPF-OT/DIMMPF-N/IMMPF`，`--experiment` 可选 `Markov/Polya/Exchange/Erlang`，并设置了学习率、`lamb`、`epochs` 等参数。

### 推荐最小实验顺序

先不要追求复现论文完整结果。

```bash
python main.py --alg IMMPF --device cpu --n_runs 1 --data_dir temp
```

然后：

```bash
python main.py --alg DIMMPF --device cpu --epochs 3 --n_runs 1 --data_dir temp
```

再做对比：

```bash
python main.py --alg RLPF --device cpu --epochs 3 --n_runs 1 --data_dir temp
python main.py --alg DIMMPF-N --device cpu --epochs 3 --n_runs 1 --data_dir temp
```

目的不是跑出好结果，而是确认：

```text
1. 数据怎么生成
2. 模型怎么初始化
3. 滤波器怎么跑
4. loss 怎么更新
5. 结果保存在哪里
```

### 训练代码要看什么

`trainingRS.py` 的 `e2e_train()` 里，训练流程包括数据划分、标准化、构造 likelihood loss 和 supervised loss、前向运行 DPF、反向传播、梯度裁剪和参数更新。

其中 `lam != 0` 时会构造：

```python
likelihood_loss = Magnitude_Loss(results.Log_Likelihood_Factors(), sign=-1)
complete_loss = Compound_Loss([loss, likelihood_loss])
```

然后用：

```python
complete_loss.register_data(weights=pt.tensor([1., lam], device='cuda'))
```

把 MSE 和 likelihood/ELBO 部分组合起来。

---

# 5. 最适合你的小论文改进方向

你现在的目标是“看懂论文和代码，然后改进算法写小论文”。不要一开始就做大改。建议从下面 3 个方向里选一个。

## 方向 A：改进切换模型 `NN_Switching`

这是最适合写小论文的方向。

原始代码中有四类切换模型：

```text
Markov_Switching
Polya_Switching
Erlang_Switching
NN_Switching
```

其中 `NN_Switching` 是 DIMMPF 的核心可学习切换模块。代码里它用 recurrent hidden state 输出 regime 概率，并通过 `get_regime_probs()` 给 IMMPF/DIMMPF 使用。

可改进点：

```text
1. 给 NN_Switching 加入观测 y_t 作为输入
2. 把简单 recurrent 结构换成 GRU/LSTM
3. 给切换概率加温度系数 temperature
4. 加入 entropy regularization，避免某个模型过早塌缩
```

小论文题目可以写：

```text
一种观测引导的可微交互多模型粒子滤波切换机制改进
```

优点：

```text
改动集中
实验容易做
和原论文核心关联强
容易解释创新点
```

---

## 方向 B：改进重采样策略

`simulationRS.py` 中 `advance_one()` 的核心就是根据 regime probability 和粒子权重为每个模型重采样。

可改进点：

```text
1. 调整每个 regime 的粒子分配数量
2. 不再固定每个模型 n_particles // n_models
3. 根据模型概率动态分配粒子
4. 低概率模型保留最小粒子数，避免模式消失
```

小论文题目可以写：

```text
面向 Regime-Switching 系统的自适应粒子分配 DIMMPF 方法
```

优点：

```text
直观
和 IMMPF 的“每模型粒子数”直接相关
实验指标容易设计
```

难点：

```text
需要认真处理张量 shape 和每个 k 的粒子数量
```

---

## 方向 C：改进损失函数

当前训练里主要是 MSE 与 likelihood/ELBO 组合，`lamb` 控制 likelihood loss 相对 MSE 的权重。

可改进点：

```text
1. 动态调整 lambda
2. 前期偏 MSE，后期增加 likelihood/ELBO
3. 加入 regime 分类辅助损失
4. 对切换频繁区间加权
```

小论文题目可以写：

```text
基于动态损失权重的 DIMMPF 训练方法研究
```

优点：

```text
代码改动较小
容易做消融实验
```

难点：

```text
理论创新性可能弱于方向 A 和 B
```

---

# 6. 我建议你选哪个方向？

我建议你选：

> **方向 A：改进 `NN_Switching`，做观测引导的 regime switching。**

理由：

```text
1. 和 DIMMPF 的核心最相关
2. 改动范围集中在 Net.py
3. 可以和原始 NN_Switching 做直接对比
4. 小论文容易讲清楚：原方法只依赖 regime hidden state，你的方法额外利用观测信息辅助判断模式切换
```

你的小论文主线可以是：

```text
问题：
    原 DIMMPF 的切换模块主要根据历史 regime 表示推断下一模式，
    对观测突变的直接利用不足。

方法：
    在 NN_Switching 中引入当前观测或观测编码，
    使 regime transition probability 同时依赖历史模式状态和观测信息。

实验：
    在 Markov / Polya / Erlang 三类切换数据上比较
    IMMPF、DIMMPF、改进 DIMMPF。

指标：
    MSE、per-step loss、训练稳定性、切换点附近误差。
```

---

# 7. 你的 4 周学习安排

## 第 1 周：恢复 + IMMPF

| 天 | 内容 | 产出 |
|---|---|---|
| 1 | 复习 PF 三函数 | 一页 PF 流程图 |
| 2 | 看 `IMMPF.M_0_proposal/M_t_proposal/log_f_t` | 解释 `k` 的含义 |
| 3 | 看 `get_regime_probs()` | 解释 regime probability |
| 4 | 看 `simulationRS.initialise()` | 写初始化流程 |
| 5 | 看 `advance_one()` 前半段 | 写重采样流程 |
| 6 | 看 `advance_one()` 后半段 | 写权重更新流程 |
| 7 | 自测 + 整理笔记 | 完成 IMMPF 笔记 |

## 第 2 周：DIMMPF

| 天 | 内容 | 产出 |
|---|---|---|
| 1 | 看 `NN_Switching` | 画 hidden state 更新图 |
| 2 | 看 `Simple_NN` | 明白网络结构 |
| 3 | 看 `DIMMPF.M_t_proposal` | 解释动态网络 |
| 4 | 看 `DIMMPF.log_f_t` | 解释观测网络 |
| 5 | 对比 IMMPF 与 DIMMPF | 写一张对比表 |
| 6 | 看 `main.py` 中 DIMMPF 初始化 | 写参数说明 |
| 7 | 运行最小实验 | 记录输出结果 |

## 第 3 周：训练与论文映射

| 天 | 内容 | 产出 |
|---|---|---|
| 1 | 看 `trainingRS.e2e_train()` | 画训练流程 |
| 2 | 看 MSE loss | 解释监督误差 |
| 3 | 看 likelihood/ELBO loss | 解释 `lambda` |
| 4 | 读论文算法部分 | 标出对应代码 |
| 5 | 读论文实验部分 | 找出 main.py 对应参数 |
| 6 | 跑 IMMPF/RLPF/DIMMPF 对比 | 保存结果 |
| 7 | 整理“论文公式到代码映射” | 形成小论文基础材料 |

## 第 4 周：确定改进点

| 天 | 内容 | 产出 |
|---|---|---|
| 1 | 选改进方向 | 明确创新点 |
| 2 | 写 baseline 实验表 | 原始结果 |
| 3 | 改 `NN_Switching` | 代码版本 1 |
| 4 | 跑 Markov 实验 | 初步结果 |
| 5 | 跑 Polya/Erlang 实验 | 对比结果 |
| 6 | 画图 | loss 曲线 / per-step error |
| 7 | 写小论文大纲 | 摘要、方法、实验、结论 |

---

# 8. 你现在立刻执行的最小任务

今天只做这 3 件事：

```text
1. 打开 IMMPF学习笔记.md
2. 打开 Net.py 的 class IMMPF
3. 只看这四个函数：
   M_0_proposal(k,...)
   M_t_proposal(k,...)
   log_f_t(k,...)
   get_regime_probs(...)
```

今天不要看：

```text
DIMMPF
DIMMPF_redefined
trainingRS.py
ELBO
论文证明
实验表格
```

今天的目标只有一个：

> **能说清楚：为什么 IMMPF 比 PF 多了一个 `k`。**

这个问题说清楚，你就重新进入状态了。

---

**Sources:**

- [Differentiable Interacting Multiple Model Particle Filtering](https://arxiv.org/abs/2410.00620)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)