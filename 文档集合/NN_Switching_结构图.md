# NN_Switching 结构图

> 对应代码：`Net.py` 第 966~1275 行（`NN_Switching` 类）
> 作用：用类 LSTM 的循环神经网络学习模型切换概率

---

## 整体架构图

```
输入状态 x_{t-1}
├─► 模型索引 k_{t-1} (标量)
│        │
│        ▼
│   [One-hot 编码] ───────► [0,0,0,1,0,0,0,0] (8维向量)
│        │                              │
│        │                              ▼
│        │                         ┌─────────────┐
│        │                         │   forget    │◄── 遗忘门(当前输入决定遗忘多少)
│        │                         │  (输入遗忘)  │
│        │                         └──────┬──────┘
│        │                                │
│        │                                ▼
│        │                    ┌───────────────────────┐
│        │                    │  h_{t-1} ⊙ forget    │◄── 历史信息经遗忘门过滤
│        │                    │     (逐元素乘)        │
│        │                    └───────────┬───────────┘
│        │                                │
│        │                                ▼
│        │                    ┌───────────────────────┐
│        │                    │  ⊙ self_forget(h)    │◄── 自遗忘门(历史自我衰减)
│        │                    │     (逐元素乘)        │
│        │                    └───────────┬───────────┘
│        │                                │
│        │                                ▼
│        │                    ┌───────────────────────┐
│        └───────────────────►│  + to_reccurrent     │◄── 加入当前模型的转换特征
│           (8维 one-hot)     │    (新输入注入)       │
│                             └───────────┬───────────┘
│                                         │
│                                         ▼
│                             新隐藏状态 h_t (32维)
│                                         │
└─────────────────────────────────────────┘
                                          │
                                          ▼
                                   [output_layer]
                                    Linear+Tanh
                                          │
                                          ▼
                                      logits (8维)
                                          │
                                          ▼
                                    [取绝对值]
                                          │
                                          ▼
                                    [归一化]
                                          │
                                          ▼
                                    true_probs
                                          │
                                          ▼
                              [软化: 混合网络预测+均匀分布]
                                          │
                                          ▼
                                    最终概率 probs
                                          │
                                          ▼
                                [pt.multinomial 采样]
                                          │
                                          ▼
                                    新模型索引 k_t

输出状态 x_t = [k_t, h_t]
```

---

## 与标准 LSTM 的对比

| 组件 | 本代码 (`NN_Switching`) | 标准 LSTM | 作用 |
|------|------------------------|-----------|------|
| `forget` | Linear(8→32) + Sigmoid | 遗忘门 | 根据**当前输入**决定遗忘多少历史 |
| `self_forget` | Linear(32→32) + Sigmoid | 无（近似循环连接） | 历史信息**自我衰减**（类似记忆淡化）|
| `to_reccurrent` | Linear(8→32) + Tanh | 输入转换 | 把 one-hot 编码映射到隐藏空间 |
| `output_layer` | Linear(32→32→8) | 输出层 | 从隐藏状态预测 8 个模型的概率 |

---

## 状态更新公式

```text
h_t = (h_{t-1} ⊙ self_forget(h_{t-1})) ⊙ forget(one_hot) + to_reccurrent(one_hot)

其中 ⊙ 表示逐元素相乘 (Hadamard积)
```

---

## 为什么要软化 (`softness`)

```text
softness=1.0:  完全相信神经网络预测
softness=0.0:  完全随机（均匀分布）
softness=0.9:  90%网络预测 + 10%随机探索  ← 默认配置，防止过早收敛到局部最优
```

类比去餐厅：
- **纯利用** (softness=1)：只点喜欢的菜 → 可能错过其他好菜
- **纯探索** (softness=0)：完全随机点菜 → 可能点到不好吃的
- **混合** (softness=0.9)：90%喜欢的 + 10%新菜 → 平衡稳定性和新鲜感

---

## 关于 `scale` 模块的说明

代码中定义了 `self.scale`（第 1051 行），但**在 `forward()` 中当前未被使用**。

```python
# __init__ 里定义了
self.scale = pt.nn.Sequential(
    pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid()
)

# 但 forward 里实际用的是：
c = old_recurrent * self.self_forget(old_recurrent)
c *= self.forget(one_hot)
c += self.to_reccurrent(one_hot)   # ← 没有 self.scale
```

**三种可能**：

| 情况 | 说明 |
|------|------|
| **代码遗漏** | 作者本来想写 `c += self.scale(one_hot) * self.to_reccurrent(one_hot)`，但漏了 |
| **历史遗留** | 早期版本用过，后来重构时忘删 |
| **预留扩展** | 为后续实验预留的接口 |

**如果补上 `scale`，公式应该长这样**：

```text
c += self.scale(one_hot) * self.to_reccurrent(one_hot)
#        ↑ 输入门           ↑ 输入转换
```

- `scale`：控制**允许多少新信息**进入（0~1）
- `to_reccurrent`：新信息本身是什么（-1~1）

当前代码少了 `scale` 这道"闸门"，新信息是**直接灌进去**的，没有经过流量控制。

---

*生成日期：2026-06-30*
