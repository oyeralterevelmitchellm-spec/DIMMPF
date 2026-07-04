# =============================================================================
# 状态空间模型和数据集定义
# 本文件定义了粒子滤波所需的基础模型类和数据加载类
# =============================================================================

from typing import Callable, Iterable, Generator, Iterator
import copy
import torch as pt
import os
import shutil
from enum import Enum
from warnings import warn
from .utils import nd_select, batched_select


# =============================================================================
# 第一部分：Feynman-Kac 模型基类
# 粒子滤波的数学基础，定义了所有粒子滤波模型需要实现的接口
# =============================================================================

class Feynman_Kac(pt.nn.Module):
    """
    Feynman-Kac 模型的抽象基类
    
    任何用于粒子滤波的模型都应该继承这个类
    
    核心概念：
    - M_0_proposal 和 M_t_proposal: 从提议分布中采样
    - G_0 和 G_t: 权重函数（点密度评估）
    - 观测值被视为模型参数，在每个时间步设置而不是作为函数参数传递
    
    参数:
    ----------
    **kwargs: any
        传递给 set_model_parameters 的模型参数
    """

    # 粒子滤波类型枚举
    class PF_Type(Enum):
        Undefined = 0   # 未定义
        Bootstrap = 1   # Bootstrap 滤波
        Guided = 2      # 引导滤波
        Auxiliary = 3   # 辅助粒子滤波

    # 重索引数组类
    # 用于访问观测值，支持自定义起始索引
    # 例如：可以设置索引从 -5 开始，访问 y[-5], y[-4], ..., y[0]
    class reindexed_array:
        """
        内部类：用于重索引数组，实现逻辑索引访问

        核心功能：让数组可以用"业务含义"（如时间t）作为索引访问，
        而不是数组实际存储位置（如索引0,1,2）。

        设计用途：
        ----------
        - 以 y[t] 的形式访问时间 t 的观测值
        - 隐藏数组实际从0开始的细节
        - 代码更直观，减少索引计算错误

        核心概念：
        ----------
        逻辑索引 vs 实际索引：
        - 实际索引：数组中元素的真实位置（0, 1, 2, ...）
        - 逻辑索引：元素代表的业务含义（t=4, t=5, ...）

        转换公式：index_actual = index_logical - base_index

        参数:
        ----------
        base_index: int
            第一个存储项的期望索引（逻辑索引的起始值）
            例如：base_index=4 表示 array[0] 对应逻辑索引 4

        ls: list
            实际存储的数组数据

        使用示例:
        ----------
        # 创建 reindexed_array，存储时间 4 和 5 的观测
        y = reindexed_array(base_index=4, ls=[y_4, y_5])

        # 内部结构
        # 实际数组: [y_4, y_5]，实际索引 0 和 1
        # 逻辑索引: base_index=4，表示 array[0] 对应逻辑索引 4

        # 访问方式（自动转换）
        y[4]  →  array[4-4] = array[0] = y_4   # 获取 y_4
        y[5]  →  array[5-4] = array[1] = y_5   # 获取 y_5

        好处：
        ----------
        - 代码直观：用 y[t] 直接访问时间 t 的观测
        - 隐藏细节：不需要关心数组实际从 0 开始存储
        - 减少错误：避免手动计算 t - start_index 的偏移

        注意事项:
        ----------
        - 创建后应视为不可变和只读
        - 不支持切片操作（只实现了 __getitem__）
        """

        def __init__(self, base_index: int, ls):
            """
            初始化重索引数组

            参数:
            ----------
            base_index: int
                逻辑索引的起始值，array[0] 对应的逻辑索引
            ls: list
                实际存储的数组数据
            """
            super().__init__()
            # 实际存储的数组，索引从 0 开始
            self.array = ls
            # 基础索引偏移量：逻辑索引 = 实际索引 + base_index
            self.base_index = base_index

        def __getitem__(self, index):
            """
            通过逻辑索引访问实际数组

            自动转换：逻辑索引 → 实际索引
            公式：实际索引 = 逻辑索引 - base_index

            参数:
            ----------
            index: int
                逻辑索引（如时间 t）

            返回:
            ----------
            实际数组中对应的元素

            示例:
            ----------
            self.base_index = 4, self.array = [y4, y5]
            index = 5 → return self.array[5-4] = self.array[1] = y5
            """
            # 通过偏移量访问实际数组
            # 逻辑索引 - base_index = 实际索引
            return self.array[index - self.base_index]

    def set_observations(self, get_observation: Callable, t: int):
        """
        设置观测值的抽象接口（子类必须实现）

        这是 Feynman-Kac 模型的抽象方法，定义了设置观测值的标准接口。
        所有继承 Feynman_Kac 的子类（如 PF、IMMPF 等）必须重写此方法。

        设计意图：
        ----------
        - 统一接口：所有粒子滤波模型都有相同的设置观测值方法
        - 多态支持：基类定义接口，子类实现具体逻辑
        - 延迟绑定：运行时根据实际对象类型调用对应实现

        参数:
        ----------
        get_observation: Callable
            获取观测值的函数，通常来自 State_Space_Object 的 _get_observation 方法
            调用方式: get_observation(t) → 返回时间 t 的观测值 (Tensor)
            观测值形状: (batch_size, observation_dimension)

        t: int
            当前时间步（逻辑索引）
            通常需要存储 t-1 和 t 两个时刻的观测值

        子类实现示例:
        ----------
        在 Net.py 中的 PF 类实现:

            def set_observations(self, get_observation: Callable, t: int):
                # 使用 reindexed_array 存储 t-1 和 t 两个时刻的观测
                self.y = self.reindexed_array(
                    t - 1,  # base_index: 第一个元素的逻辑索引
                    [get_observation(t - 1), get_observation(t)]  # 实际存储的数组
                )
                # 结果: self.y[t-1] 获取 y_{t-1}, self.y[t] 获取 y_t

        继承关系:
        ----------
        Feynman_Kac (基类，定义抽象接口)
            ↓ 继承
        SSM (状态空间模型)
            ↓ 继承
        HMM (隐马尔可夫模型)
            ↓ 继承
        PF/IMMPF/... (具体实现类，在 Net.py 中实现此方法)

        注意事项:
        ----------
        - 基类方法抛出 NotImplementedError，强制子类必须实现
        - 子类通常使用 reindexed_array 来存储观测值，支持逻辑索引访问
        - 观测值通常在粒子滤波的每个时间步调用一次此方法设置

        Raises:
        ----------
        NotImplementedError: 如果子类没有重写此方法
        """
        NotImplementedError("Function to set observations not implemented")

    def to(self, **kwargs):
        """
        将模型移动到指定设备和数据类型

        参数:
        ----------
        **kwargs: 关键字参数
            - device: str, 目标设备 ('cuda' 或 'cpu')
            - dtype: torch.dtype, 目标数据类型 (如 torch.float32)

        详细说明:
        ----------
        执行流程:
        1. 更新 self.device 属性
        2. 遍历所有属性，移动非 Parameter 的 Tensor
        3. 调用父类方法处理 Parameter 和子模块

        为什么需要额外处理:
        - super().to() 只处理 Parameter 和 buffer
        - 模型可能有普通 Tensor 属性（如预计算矩阵）
        - 本代码确保所有 Tensor 都移动到正确设备

        示例:
        ----------
        model.to(device="cuda", dtype=torch.float32)
        # 将所有 Tensor 和 Parameter 移动到 GPU，并转为 float32
        """
        # 第1步: 更新设备属性
        if kwargs["device"] is not None:
            self.device = kwargs["device"]

        # 第2步: 遍历所有属性，移动非 Parameter 的 Tensor
        # vars(self): 获取对象的所有属性字典
        # isinstance(var, pt.Tensor): 检查是否是 PyTorch 张量
        # not isinstance(var, pt.nn.Parameter): 排除模型参数（由父类处理）
        for var in vars(self):
            if isinstance(var, pt.Tensor) and not isinstance(var, pt.nn.Parameter):
                var.to(dtype=kwargs["dtype"], device=kwargs["device"])

        # 第3步: 调用父类 nn.Module 的 to 方法
        # 处理模型参数 (Parameter) 和子模块
        super().to(**kwargs)

    def __init__(self, device: str = "cuda") -> None:
        """
        初始化 Feynman-Kac 模型

        参数:
        ----------
        device: str
            运行设备，'cuda' 或 'cpu'

        初始化流程:
        1. 调用父类 nn.Module 初始化
        2. 设置算法类型为 Undefined（子类会覆盖）
        3. 保存设备信息
        4. 创建随机数生成器（CUDA需指定device，CPU用默认）
        """
        super().__init__()
        # 算法类型：Undefined(0), Bootstrap(1), Guided(2), Auxiliary(3)
        # 子类在初始化时会设置为具体类型
        self.alg = self.PF_Type.Undefined
        # 保存运行设备
        self.device = device
        # 创建随机数生成器
        # 注意：CPU不支持Generator(device=...)，需区分处理
        if device == "cuda":
            self.rng = pt.Generator(device=self.device)  # GPU随机数生成器
        else:
            self.rng = pt.Generator()  # CPU默认生成器

    # 评估 G_0: 时间 0 的权重函数
    def log_G_0(self, x_0):
        """
        时间 0 的权重函数（对数形式，抽象方法）

        G_0(x_0) = p(y_0 | x_0)，给定初始状态x_0的观测概率
        取对数避免数值下溢：log p(y_0|x_0)

        参数:
        ----------
        x_0: Tensor, 初始状态，形状 (batches, n_samples, state_dim)

        返回:
        ----------
        Tensor: 对数权重，形状 (batches, n_samples)
        """
        NotImplementedError("Weighting function not implemented for time 0")

    # 采样 M_0: 时间 0 的提议分布
    def M_0_proposal(self, batches: int, n_samples: int):
        """
        时间 0 的提议分布采样（抽象方法）

        从提议分布中采样初始粒子，是粒子滤波的第一步。

        参数:
        ----------
        batches: int - 批量大小
        n_samples: int - 每批粒子数

        返回:
        ----------
        Tensor: 初始粒子，形状 (batches, n_samples, state_dim)

        与log_G_0的关系:
        M_0_proposal → 生成初始粒子 → log_G_0 → 计算对数权重
        """
        NotImplementedError("Proposal model sampler not implemented for time 0")

    def log_M_0(self, x_0):
        """时间 0 的提议分布密度（对数形式）"""
        NotImplementedError("Proposal density not implemented for time 0")

    # 评估 G_t: 时间 t 的权重函数
    def log_G_t(self, x_t, x_t_1, t: int):
        """时间 t 的权重函数（对数形式）"""
        NotImplementedError("Weighting function not implemented for time t")

    # 采样 M_t: 时间 t 的提议分布
    def M_t_proposal(self, x_t_1, t: int):
        """时间 t 的提议分布采样"""
        NotImplementedError("Proposal model sampler not implemented for time t")

    def log_M_t(self, x_t, x_t_1, t: int):
        """时间 t 的提议分布密度（对数形式）"""
        NotImplementedError("Proposal density not implemented for time t")

    def observation_generation(self, x_t):
        """从状态生成观测值"""
        raise NotImplementedError("Observation generation not implemented")

        # 时间 0 (初始化):
#     M_0_proposal() ──► 生成 x_0 ──► log_G_0(x_0) ──► 计算初始权重
#            │                              │
#            ▼                              ▼
#     log_M_0(x_0) ◄────────────────── 权重修正（如需要）

# 时间 t (迭代):
#     x_{t-1} ──► M_t_proposal() ──► 生成 x_t ──► log_G_t() ──► 计算权重
#                    │                                      │
#                    ▼                                      ▼
#             log_M_t() ◄──────────────────────────── 权重修正

# NotImplementedError("...") 创建异常（没效果）；
# raise NotImplementedError("...") 创建并抛出异常（正确用法）。raise 是关键，没有它异常不会生效。

# =============================================================================
# 第二部分：状态空间模型 (SSM)
# 辅助 Feynman-Kac 模型的基类
# =============================================================================

class SSM(Feynman_Kac):
    """
    状态空间模型的基类
    
    注意:
    ------
    R_t 是 Radon-Nikodym 导数 M_t(x_t-1, dx_t) / P_t(x_t-1, dx_t)
    标准情况下应该是可计算的
    
    提供了计算辅助权重函数 G_t 的标准形式
    但为了性能或计算可行性，可以直接覆盖计算
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device)
        self.PF_type = "Auxiliary"

    def log_R_0(self, x_0):
        """
        时间 0 的动态/提议 Radon-Nikodym 导数（抽象方法）

        R_0 = M_0(x_0) / P_0(x_0)，提议分布与动态分布的比值
        取对数形式：log M_0 - log P_0

        用于纠正提议分布和真实状态转移分布的差异。

        参数:
        ----------
        x_0: Tensor, 初始状态，形状 (batches, n_samples, state_dim)

        返回:
        ----------
        Tensor: 对数导数值，形状 (batches, n_samples)
        """
        raise NotImplementedError(
            "Dynamic/Proposal Radon-Nikodym derivative not implemented for time zero"
        )

    def log_R_t(self, x_t, x_t_1, t: int):
        """
        时间 t 的动态/提议 Radon-Nikodym 导数（抽象方法）

        R_t = M_t(x_{t-1}, x_t) / P_t(x_{t-1}, x_t)
        提议分布与真实状态转移分布的比值。

        参数:
        ----------
        x_t: Tensor, 当前状态
        x_t_1: Tensor, 上一时刻状态
        t: int, 时间步

        返回:
        ----------
        Tensor: 对数导数值
        """
        raise NotImplementedError(
            "Dynamic/Proposal Radon-Nikodym derivative not implemented for time t"
        )

    def log_f_t(self, x_t, t: int):
        """
        观测似然函数（抽象方法）

        f_t(x_t) = p(y_t | x_t)，给定状态下观测到 y_t 的概率。
        取对数形式避免数值下溢。

        示例:
            x_t = [1.0, 0.5], y_t = [1.2, 0.6]
            log_f_t = log N(y_t; x_t, σ²)  # 假设高斯噪声

        参数:
        ----------
        x_t: Tensor, 当前状态
        t: int, 时间步

        返回:
        ----------
        Tensor: 对数似然值
        """
        raise NotImplementedError("Observation likelihood not implemented")

    def log_eta_t(self, x_t, t: int):
        """
        辅助权重函数（抽象方法）

        用于辅助粒子滤波（APF）的权重调整。
        η_t 帮助粒子更好地逼近真实后验分布。

        标准权重: w_t ∝ p(y_t|x_t) * p(x_t|x_{t-1}) / q(x_t|x_{t-1})
        辅助权重: w_t ∝ ... * η_t

        参数:
        ----------
        x_t: Tensor, 当前状态
        t: int, 时间步

        返回:
        ----------
        Tensor: 辅助权重的对数值
        """
        raise NotImplementedError("Auxililary weights not implemented")

# 完整权重 log_G_t = 引导权重 + 辅助调整
#                     │
#                     ├──► log_G_t_guided = log_R_t + log_f_t
#                     │                    │
#                     │                    ├──► log_R_t: 提议-动态比率
#                     │                    └──► log_f_t: 观测似然
#                     │
#                     └──► 辅助调整 = log_eta_t(x_t) - log_eta_t(x_{t-1})

    def log_G_0_guided(self, x_0):
        """引导滤波的时间 0 权重"""
        return self.log_R_0(x_0) + self.log_f_t(x_0, 0)

    def log_G_t_guided(self, x_t, x_t_1, t: int):
        """引导滤波的时间 t 权重"""
        return self.log_R_t(x_t, x_t_1, t) + self.log_f_t(x_t, t)

    def log_G_0(self, x_0):
        """时间 0 的完整权重（辅助形式）"""
        return self.log_G_0_guided(x_0) + self.log_eta_t(x_0, 0)

    def log_G_t(self, x_t, x_t_1, t: int):
        """时间 t 的完整权重（辅助形式）"""
        return (
            self.log_G_t_guided(x_t, x_t_1, t)
            + self.log_eta_t(x_t, t)
            - self.log_eta_t(x_t_1, t - 1)
        )


# =============================================================================
# 第三部分：隐马尔可夫模型 (HMM)
# 实现了具体的采样方法
# =============================================================================

class HMM(SSM):
    """
    隐马尔可夫模型
    实现了具体的状态生成和提议采样方法
    """

    def generate_state_0(self):
        """
        生成时间 0 的候选状态（抽象方法）

        生成 S 个候选状态，用于 M_0_proposal 的采样。
        子类根据具体的状态空间模型实现。

        返回:
        ----------
        Tensor: 候选状态，形状 (S, D)
                S = 候选数量，D = 状态维度

        示例:
            state = [[1.0, 0.5, -0.3, 0.8],   # 候选1
                     [0.9, 0.6, -0.2, 0.7],   # 候选2
                     ...,
                     [1.1, 0.4, -0.4, 0.9]]   # 候选S
        """
        raise NotImplementedError("State generation not implemented for time 0")

    def M_0_proposal(self, batches: int, n_samples: int):
        """
        时间 0 的提议分布采样

        Bootstrap 粒子滤波的第一步：生成初始粒子。

        工作流程:
        1. 生成候选状态 (generate_state_0)
           - 生成 S 个候选状态，形状 (S, D)

        2. 计算每个候选的概率 (log_M_0)
           - state.unsqueeze(0): (S,D) → (1,S,D)
           - log_M_0: 计算对数概率，(1,S,D) → (1,S)
           - squeeze(): (1,S) → (S,)
           - 结果: probs = [-2.3, -1.8, ...]，100个对数概率

        3. 使用多项式采样选择 (multinomial)
           - pt.exp(probs): 对数概率 → 概率
           - multinomial(..., batches*n_samples, True): 按概率采样
           - reshape(batches, n_samples): 重塑为 (B, N)
           - 结果: indices = [[1,0,1], [4,1,2]]，采样索引

        4. 根据索引选择状态 (nd_select)
           - 从候选状态中选择，返回 (B, N, D)

        完整流程示例 (batches=2, n_samples=3):
        ----------
        state (100,4) → probs (100,) → indices (2,3) → 返回 (2,3,4)

        参数:
        ----------
        batches: int - 批量大小
        n_samples: int - 每批粒子数

        返回:
        ----------
        Tensor: 初始粒子，形状 (batches, n_samples, state_dim)

        与 Bootstrap 滤波的关系:
        ----------
        x_0 = model.M_0_proposal(batches, n_samples)  # ← 本方法
        w_0 = model.log_G_0(x_0)  # 计算初始权重
        # 然后进入 M_t_proposal 和 log_G_t 的循环
        """
        # 第1步: 生成候选状态
        # generate_state_0 生成 S 个候选，形状 (S, D)
        state = self.generate_state_0()  # SxD

        # 第2步: 计算每个候选的对数概率
        # unsqueeze(0): 增加 batch 维度，(S,D) → (1,S,D)
        # log_M_0: 计算对数概率
        # squeeze(): 去掉维度1，(1,S) → (S,)
        probs = self.log_M_0(state.unsqueeze(0)).squeeze()  # S

        # 第3步: 按概率多项式采样
        # pt.exp(probs): 对数概率 → 概率
        # multinomial(..., batches*n_samples, True): 采样 batches*n_samples 个，有放回
        # reshape(batches, n_samples): 重塑为 (B, N)
        indices = pt.multinomial(pt.exp(probs), batches * n_samples, True).reshape(
            batches, n_samples
        )  # BxN

        # 第4步: 根据索引选择状态
        # nd_select: 从 state 中选择 indices 对应的状态
        # 返回形状: (B, N, D)
        return nd_select(state, indices)  # BxNxD

    def generate_state_t(self, x_t_1, t: int):
        """
        生成时间 t 的候选状态（抽象方法）

        基于前一时刻状态 x_{t-1} 生成 S 个候选状态。
        用于 M_t_proposal 的采样。

        参数:
        ----------
        x_t_1: Tensor, 上一时刻状态，形状 (B, N, D)
        t: int, 时间步

        返回:
        ----------
        Tensor: 候选状态，形状 (B, N, S, D)
                每个粒子有 S 个候选
        """
        raise NotImplementedError("State generation not implemented for time t")

    def M_t_proposal(self, x_t_1, t: int):
        """
        时间 t 的提议分布采样

        Bootstrap 粒子滤波中 t>0 时的提议采样。
        与 M_0_proposal 的区别：需要前一时刻状态 x_{t-1}。

        工作流程:
        1. 基于前一时刻状态生成候选 (generate_state_t)
           - 输入: x_t_1 (B,N,D)
           - 输出: state (B,N,S,D)，每个粒子 S 个候选

        2. 计算每个候选的概率 (log_M_t)
           - 输入: state, x_t_1, t
           - 输出: probs (B,N,S)，每个候选一个对数概率

        3. 使用多项式采样选择 (multinomial)
           - pt.flatten(probs, 0, 1): (B,N,S) → (B*N, S)
           - multinomial(..., 1, True): 每行采样 1 个，(B*N, 1)
           - reshape(B, N): (B*N, 1) → (B, N)
           - 结果: indices (B,N)，每个粒子选 1 个候选

        4. 根据索引选择状态 (batched_select)
           - 从 state (B,N,S,D) 中选择 indices (B,N)
           - 结果: (B, N, D)

        完整流程示例 (B=2, N=3, S=5, D=4):
        ----------
        x_t_1 (2,3,4) → state (2,3,5,4) → probs (2,3,5)
            → flatten(6,5) → multinomial(6,1) → reshape(2,3)
            → indices (2,3) → batched_select → 返回 (2,3,4)

        参数:
        ----------
        x_t_1: Tensor, 上一时刻状态，形状 (B, N, D)
        t: int, 当前时间步

        返回:
        ----------
        Tensor: 新时刻粒子，形状 (B, N, D)

        与 M_0_proposal 的区别:
        ----------
        | 特点 | M_0_proposal | M_t_proposal |
        |------|-------------|--------------|
        | 时间 | t=0 | t>0 |
        | 输入 | 无 | x_{t-1} |
        | 候选形状 | (S,D) | (B,N,S,D) |
        | 采样 | 从 S 中选 | 每粒子从 S 中选 |

        在粒子滤波中的作用:
        ----------
        for t in range(1, T):
            x_t = model.M_t_proposal(x_{t-1}, t)  # ← 本方法
            w_t = model.log_G_t(x_t, x_{t-1}, t)  # 计算权重
            x_t = resample(x_t, w_t)  # 重采样
        """
        # 第1步: 基于 x_{t-1} 生成候选状态
        # 每个粒子生成 S 个候选，形状 (B, N, S, D)
        state = self.generate_state_t(x_t_1, t)  # BxNxSxD

        # 第2步: 计算每个候选的对数概率
        # probs 形状: (B, N, S)
        probs = self.log_M_t(state, x_t_1, t)  # BxNxS

        # 第3步: 按概率多项式采样
        # flatten(0,1): (B,N,S) → (B*N, S)，合并批次和粒子维度
        # multinomial(..., 1, True): 每行采样 1 个，形状 (B*N, 1)
        # reshape(B, N): 重塑为 (B, N)
        indices = pt.multinomial(pt.flatten(pt.exp(probs), 0, 1), 1, True).reshape(
            x_t_1.shape(0), x_t_1.shape(1)
        )  # BxN

        # 第4步: 根据索引选择状态
        # batched_select: 从 state (B,N,S,D) 中选择 indices (B,N)
        # 返回形状: (B, N, D)
        return batched_select(state, indices)  # BxNxD


# =============================================================================
# 第四部分：状态空间对象
# 用于生成和管理观测数据
# =============================================================================

class State_Space_Object:
    """
    通用状态空间对象的基类

    抽象基类，定义获取观测值的接口。子类实现具体逻辑：
    - Simulated_Object: 动态生成观测数据
    - Observation_Queue: 从文件/张量加载观测

    核心机制:
    ----------
    1. 观测历史管理: 维护列表存储连续时间步的观测
       示例: 列表[y_5,y_6,y_7], time_index=5, 访问y_7 → list[7-5]=list[2]

    2. 懒加载: 请求观测时，存在则返回，不存在则推进状态生成
       示例: 当前到t=3，请求y_5 → 生成y_4,y_5后返回

    参数:
    -----------
    observation_history_length: int - 循环缓冲区大小
    observation_dimension: int - 观测维度

    注意:
    --------
    - 不建议在创建和_get_observation()之外使用
    - 状态转移和观测生成使用不同RNG，保证可重复性
    """

    def _get_observation(self, t):
        """获取时间t的观测值（抽象方法，子类实现）"""
        # pass: Python空操作占位符，定义抽象接口，强制子类实现
        pass

    def save(self):
        """保存数据（抽象方法，子类实现）"""
        pass


# =============================================================================
# 第五部分：模拟对象
# 用于生成模拟的隐马尔可夫过程数据
# =============================================================================

class Simulated_Object(State_Space_Object):
    """
    模拟对象的基类
    
    此类模拟一个隐马尔可夫过程
    为了输出可解释，给定的模型应该始终是 Bootstrap 类型
    无论将使用什么算法进行滤波
    
    不存储过去的状态，如果需要状态则应在每次调用 _get_observation 后记录
    
    参数:
    ----------
    Model: Feynman_Kac
        用于模拟的模型，必须是 Bootstrap 类型
    
    Batches: int
        并行模拟的轨迹数量
    
    observation_history_length: int
        内存中存储的最小观测值数量
    
    observation_dimension: int
        观测将具有的维度数
    """

    def __copy__(self):
        """
        复制对象
        创建一个新的状态空间对象，具有相同的 RNG 种子
        以便再次运行会产生一致的结果

        详细解释：
        -----------
        这是 Python 的 copy 模块在调用 copy.copy(obj) 时会调用的特殊方法，
        用于实现对象的浅拷贝。

        逐行解析：
        1. cls = self.__class__: 获取当前对象的类
        2. out = cls.__new__(cls): 不调用 __init__，直接创建空实例
        3. out.__dict__.update(self.__dict__): 复制所有属性到新的实例
        4. out.observations = pt.empty_like(...): 重新分配观测值存储空间（避免共享内存）
        5. out.time_index = 0: 重置时间索引
        6. out.object_time = 0: 重置对象时间
        7. out.first_object_set = False: 重置首次设置标志
        8. out.model = copy.copy(out.model): 浅拷贝模型对象
        9. out.x_t = out.model.M_0_proposal(...): 重新初始化状态

        关键设计意图：
        - 使用 __new__ 而不是直接实例化，避免触发 __init__ 中的初始化逻辑
        - 重置时间相关属性，让复制的对象从头开始模拟
        - 重新分配 observations，因为 PyTorch Tensor 是引用类型，不重新分配会导致新旧对象共享内存
        - 从模型的初始提议分布 M_0_proposal 重新采样初始状态

        RNG 种子说明：
        --------------
        RNG = Random Number Generator（随机数生成器）
        种子（Seed）是随机数生成算法的起始值。

        计算机中的"随机数"实际上是伪随机数——通过确定性算法生成，只是看起来随机。
        相同种子 → 相同随机数序列（可重复性）
        不同种子 → 不同随机数序列（多样性）

        在本代码中，"相同的 RNG 种子"意味着：
        复制对象从头开始运行，会产生与原始对象完全相同的随机序列。
        这在粒子滤波和状态空间模型中非常重要，用于：
        1. 可重复实验：科学研究需要结果可复现
        2. 对比实验：比较不同算法时，需要用相同的随机数据
        3. 调试：固定的随机序列便于定位问题

        使用场景：
        - 并行模拟：创建多个独立的模拟器副本
        - 可重复实验：确保相同的 RNG 种子产生相同的结果
        - 粒子滤波算法：在 Bootstrap Particle Filter 等算法中需要多个轨迹副本
        """
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        out.observations = pt.empty_like(out.observations, device=self.device)
        out.time_index = 0
        out.object_time = 0
        out.first_object_set = False
        out.model = copy.copy(out.model)
        out.x_t = out.model.M_0_proposal(out.batch_size, 1)
        return out

    def __init__(
        self,
        model: Feynman_Kac,
        batch_size: int,
        observation_history_length: int,
        observation_dimension: int,
        device: str = "cuda",
    ):
        """
        初始化模拟对象
        
        参数:
        ----------
        model: Feynman_Kac
            用于生成数据的模型
        batch_size: int
            批量大小（并行生成的序列数）
        observation_history_length: int
            观测历史长度
        observation_dimension: int
            观测维度
        device: str
            运行设备
        """
        self.device = device
        self.observation_history_length = observation_history_length
        self.observation_dimension = observation_dimension
        # 预分配观测值存储空间（两倍历史长度用于循环缓冲）
        self.observations = pt.empty(
            (
                batch_size,
                self.observation_history_length * 2,
                self.observation_dimension,
            ),
            device=self.device,
        )
        self.first_object_set = False
        self.time_index = 0          # 当前存储的起始时间索引
        self.object_time = 0         # 对象的当前时间
        self.model = model
        self.batch_size = batch_size
        # 初始化状态：从 M_0 提议分布采样
        self.x_t = self.model.M_0_proposal(batch_size, 1)

    def _forward(self):
        """前向推进：状态转移"""
        self.object_time += 1
        self.x_t = self.model.M_t_proposal(self.x_t, self.object_time)

    def _set_observation(self, t: int, value: pt.Tensor) -> None:
        """
        用新的观测值更新观测历史

        如果观测历史已满，将后半部分复制到前半部分
        并从中间点开始填充

        详细解释：
        -----------
        这是一个用于管理观测历史的方法，采用循环缓冲区 (Circular Buffer) 策略来高效地存储观测数据。

        核心概念：循环缓冲区
        --------------------
        observations 数组的容量为 2 * observation_history_length，分为前半部分和后半部分：
        - 前半部分：索引 0 ~ observation_history_length-1
        - 后半部分：索引 observation_history_length ~ 2*observation_history_length-1

        当缓冲区满时（后半部分也写满），将后半部分移到前半部分，清空后半部分继续使用。
        这样只需保留 2 * observation_history_length 的数据，而非无限增长。

        示例（observation_history_length = 3）：
        初始状态: [空,空,空 | 空,空,空]  time_index=0
        写入t=0,1,2: [O0,O1,O2 | 空,空,空]
        写入t=3,4,5: [O0,O1,O2 | O3,O4,O5]  ← 缓冲区满
        写入t=6触发循环: [O3,O4,O5 | 空,空,空]  time_index=3

        参数:
        ----------
        t: int
            新观测值的绝对时间步（全局时间）

        value: Tensor
            新观测值的值，形状通常是 (batch_size, 1, observation_dimension)

        代码逻辑详解：
        --------------
        1. 条件判断：if self.time_index + self.observation_history_length * 2 <= t
           检查当前缓冲区是否已满。当 t 超过 time_index + 2*容量 时触发循环。

        2. 循环缓冲操作：
           self.observations[:, :observation_history_length] = self.observations[:, observation_history_length:]
           将后半部分（较新的数据）复制到前半部分
           self.time_index += self.observation_history_length
           更新 time_index，表示缓冲区起始时间后移

        3. 写入新观测值：
           self.observations[:, t - self.time_index, :] = value.squeeze(1)
           t - time_index：将绝对时间转换为相对索引
           value.squeeze(1)：去掉维度为1的轴（中间的时间维度）

        设计优势：
        ----------
        - 内存效率：只保留固定长度的数据，而非无限增长
        - 时间效率：循环缓冲避免了频繁的数据搬移，只需一次切片复制
        - 局部性：保留最近的历史，旧的观测值被覆盖（符合时间序列特性）

        使用场景：
        ----------
        这种设计在粒子滤波和在线学习场景中非常常见，因为不需要存储无限长的历史，
        只需保留最近的观测即可进行状态估计。
        """
        if self.time_index + self.observation_history_length * 2 <= t:
            # 循环缓冲：将后半部分移到前半部分
            self.observations[:, : self.observation_history_length] = self.observations[
                :, self.observation_history_length :
            ]
            self.time_index += self.observation_history_length
        self.observations[:, t - self.time_index, :] = value.squeeze(1)

    def _get_observation(self, t):
        """
        获取时间 t 的观测值

        如果尚未创建则推进对象状态并生成观测值直到时间 t

        详细解释：
        -----------
        这是一个按需生成和获取观测值的方法，实现了惰性计算 (Lazy Evaluation) 策略——
        只在需要时才生成观测数据。

        方法整体流程：
        1. t < 0? → 返回 NaN（负时间在物理上无意义）
        2. t < time_index? → 抛出错误（数据已被循环缓冲区覆盖）
        3. t == 0 且首次访问? → 初始化并返回
        4. 否则 → 循环生成直到时间 t，然后返回

        参数:
        ----------
        t: int
            要获取观测值的时间步（绝对时间，全局）

        返回:
        ----------
        Tensor: 时间 t 的观测值，形状为 (batch_size, observation_dimension)

        代码逻辑详解：
        --------------

        1. 处理负时间（t < 0）：
           返回全 NaN 的张量，形状为 (batch_size, observation_dimension)。
           原因：负时间在物理上通常没有意义，NaN 可以清楚地标识"无效/不存在的数据"，
           避免调用者因访问未初始化的数据而产生错误结果。

        2. 检查数据是否已被覆盖（t < self.time_index）：
           由于使用了循环缓冲区（在 _set_observation 中实现），旧数据会被新数据覆盖。
           time_index 表示缓冲区中最早保存的数据的时间。
           如果请求的时间早于 time_index，说明数据已丢失，抛出 ValueError。

           示例：
           time_index = 100  ← 缓冲区已循环，最早只保存了 t=100 之后的数据
           请求 t = 50       ← 这个数据已经被覆盖了！→ 抛出错误

        3. 处理首次访问 t=0（t == 0 and not self.first_object_set）：
           时间 0 是初始状态，需要特殊初始化。
           self.x_t 在 __init__ 中已通过 M_0_proposal 初始化。
           first_object_set 标志确保只初始化一次。

           流程：
           x_t (已初始化) → observation_generation(x_t) → 生成观测值 → 存入缓冲区

        4. 按需生成观测值（while t > self.object_time）：
           这是惰性计算的核心机制。如果请求的时间 t 大于当前对象时间 object_time，
           则循环执行状态转移和观测生成，直到达到时间 t。

           示例（当前 object_time=3，请求 t=6）：
           while 6 > 3: _forward() → object_time=4, x_t更新; _set_observation(4, ...)
           while 6 > 4: _forward() → object_time=5; _set_observation(5, ...)
           while 6 > 5: _forward() → object_time=6; _set_observation(6, ...)
           while 6 > 6: 不满足，退出循环

        5. 返回观测值：
           return self.observations[:, t - self.time_index]
           t - time_index：将绝对时间转换为缓冲区中的相对索引

        与其他方法的关系：
        ------------------
        _get_observation (本方法)
            ├── _forward()              - 状态转移：x_t → x_{t+1}
            ├── _set_observation()      - 存入循环缓冲区
            └── model.observation_generation() - 从状态生成观测值

        设计优势：
        ----------
        - 惰性计算：只在需要时生成数据，避免不必要的计算
        - 统一接口：无论数据是否已存在，调用方式相同
        - 自动扩展：自动处理时间推进，调用者无需关心
        - 内存安全：明确处理数据被覆盖的情况，避免静默错误
        - 边界安全：负时间返回 NaN，过期数据抛出错误

        使用场景示例：
        --------------
        # 场景 1：顺序访问（最常见）
        for t in range(100):
            obs = sim._get_observation(t)  # 每次生成下一个

        # 场景 2：随机访问（跳转到未来）
        obs_t50 = sim._get_observation(50)   # 生成 0~50
        obs_t100 = sim._get_observation(100) # 继续生成 51~100

        # 场景 3：重复访问（已存在的数据）
        obs_t50_again = sim._get_observation(50)  # 直接返回，不重新生成

        # 场景 4：错误访问（数据已丢失）
        obs_t10 = sim._get_observation(10)  # 假设 time_index 已变为 100
        # → 抛出 ValueError: 最早存储的是 t=100

        注意：
        -----
        这个方法是隐马尔可夫模型模拟器的核心接口，封装了状态转移、观测生成和
        缓冲区管理的复杂逻辑，为上层提供简洁的观测值访问方式。
        """
        if t < 0:
            # 负时间返回 NaN
            return pt.full(
                (self.batch_size, self.observation_dimension),
                pt.nan,
                device=self.device,
            )

        if t < self.time_index:
            # time_index = 当前缓冲区的起始时间
            # 如果请求的时间早于 time_index，说明数据已被循环缓冲区覆盖
            raise ValueError(
                f"Trying to access observation at time {t}, "
                f"the earliest stored is at time {self.time_index}"
            )

        if t == 0 and not self.first_object_set:
            # 时间 0 的首次设置
            # x_t 已在 __init__ 中通过 M_0_proposal 初始化
            self.first_object_set = True
            self._set_observation(0, self.model.observation_generation(self.x_t))
            return self.observations[:, 0]

        # 生成直到时间 t 的所有观测值
        # 惰性计算：只在需要时生成，通过 while 循环逐步推进到目标时间
        while t > self.object_time:
            self._forward()  # 状态转移：x_t → x_{t+1}
            self._set_observation(
                self.object_time, self.model.observation_generation(self.x_t)
            )

        # 返回观测值，将绝对时间转换为缓冲区相对索引
        return self.observations[:, t - self.time_index]

    def save(
        self,
        path: str,
        T: int,
        quantity: int,
        prefix: str = "str",
        clear_folder=True,
        bypass_ask=False,
    ):
        """
        保存模拟数据到文件

        这是 Simulated_Object 类的数据持久化接口，用于批量生成模拟数据并保存到文件系统。

        核心设计目标：
        1. 可重复性：通过 copy.copy(self) 确保每个序列使用相同的 RNG 种子
        2. 安全性：提供文件夹清空确认机制，防止误删
        3. 批量处理：支持一次性生成大量序列数据
        4. 模块化：委托 Observation_Queue 处理具体保存逻辑

        方法整体流程：
        1. 验证模型类型（警告非 Bootstrap 模型）
        2. 文件夹管理（清空确认、删除、重建）
        3. 批量生成并保存数据

        参数:
        ----------
        path: str
            数据保存路径
        T: int
            每个序列的时间步数（序列长度）
        quantity: int
            要生成的序列数量
        prefix: str
            文件名前缀，默认为 "str"
        clear_folder: bool
            是否清空目标文件夹，默认为 True
        bypass_ask: bool
            是否跳过确认提示，默认为 False

        注意事项:
        ----------
        - Bootstrap 模型是粒子滤波的基础算法，保证模拟数据的标准性
        - 使用 warnings.warn 发出警告而非报错，允许非 Bootstrap 模型继续执行
        - 每个序列通过 copy.copy(self) 创建独立副本，确保隔离性和可重复性
        """
        # 检查模型类型
        # Bootstrap Particle Filter 特点：
        # - 提议分布 = 状态转移分布（最简单形式）
        # - 适用于标准状态空间模型
        # - 模拟数据应具有可解释性
        if self.model.alg != self.model.PF_Type.Bootstrap:
            warn(
                f"Model is {self.model.alg.name} instead of Bootstrap, are you this is right?"
            )

        # 文件夹管理：安全的文件夹清空机制
        # 包含用户确认和异常处理，防止误删重要数据
        if clear_folder:
            if os.path.exists(path):
                # 根据 bypass_ask 决定是否跳过用户确认
                if bypass_ask:
                    response = "Y"  # 自动确认，用于批量/自动化运行
                else:
                    # 交互式确认，防止误删
                    print(f"Warning: This will overwrite the directory at path {path}")
                    response = input("Input Y to confirm you want to do this:")

                # 检查用户输入，大小写兼容（"Y" 或 "y"）
                if response != "Y" and response != "y":
                    print("Halting")  # 用户取消操作
                    return  # 优雅退出，不抛异常

                # 执行删除操作，带异常处理
                try:
                    shutil.rmtree(path)  # 递归删除目录及其所有内容
                except:
                    # 备选方案：rmtree 失败时尝试删除单个文件
                    # 可能原因：path 是文件而非目录、权限不足、目录被占用
                    os.remove(path)

            # 重建目录，确保保存路径存在
            os.mkdir(path)

        # 批量生成并保存数据
        # 核心逻辑：循环 quantity 次，每次创建一个独立副本生成序列
        for i in range(quantity):
            # 创建独立副本（相同 RNG 种子，从头开始）
            # 优势：
            # 1. 隔离性：每个序列使用独立对象，避免状态干扰
            # 2. 可重复性：相同 RNG 种子确保序列生成的确定性
            # 3. 并行潜力：未来可改造为真正的并行生成
            temp = copy.copy(self)

            # 使用 Observation_Queue 封装观测序列并保存
            # i * self.batch_size：确保多批次数据的索引连续性
            # 例如：batch_size=100 时，第0批索引0-99，第1批索引100-199，以此类推
            Observation_Queue(
                conversion_object=temp, time_length=T, device=self.device
            ).save(path, i * self.batch_size, prefix, False)


# =============================================================================
# 第六部分：观测队列
# 作为观测值队列的状态空间对象
# =============================================================================

class Observation_Queue(State_Space_Object):
    """
    状态空间对象，充当观测值队列（可选状态向量）
    以简化方式重新实现某些方法，为此特殊情况提高效率
    
    参数:
    ----------
    xs: (T,s) ndarray 或 None, 默认: None
        包含维度 s 的状态的数组，在每个时间 [0,T]
        如果为 None 且 ys 不为 None，则不存储观测值
        如果 ys 为 None 则无效果
    
    ys: (T, o) ndarray 或 None, 默认: None
        包含维度 o 的观测值的数组，在每个时间 [0,T]
        如果为 None 则从 State_Space_Object conversion_object 生成观测值
    
    conversion_object: State_Space_Object, 默认: None
        要将其观测值和状态（如果可用）记忆为新的 Observation_Queue 对象的状态空间对象
        如果 ys 为 None 则必须不为 None
        否则使用 ys 加载观测值优先
    
    time_length: int 或 None, 默认: None
        要记忆的 conversion_object 的时间步数
        如果 conversion_object 为 None 则无效果
        如果 conversion_object 不为 None 则必须不为 None
    """

    def __init__(
        self,
        xs: pt.Tensor = None,
        ys: pt.Tensor = None,
        conversion_object: Simulated_Object = None,
        time_length: int = None,
        device: str = "cuda",
    ):
        """
        初始化观测队列
        
        可以直接从张量加载，或从 Simulated_Object 转换
        """
        self.device = device
        self.object_time = 0
        
        # 直接从张量加载
        if ys is not None:
            self.observations = ys
            if xs is not None:
                self.state = xs
            return

        # 从 Simulated_Object 转换
        try:
            state_dim = conversion_object.x_t.size()
            self.state = pt.empty(
                (state_dim[0], time_length + 1, state_dim[-1]), device=self.device
            )
            state_availiable = True
        except AttributeError:
            state_availiable = False

        # 禁用梯度计算以提高效率
        with pt.inference_mode():
            for t in range(time_length + 1):
                if t == 0:
                    o0 = conversion_object._get_observation(0)
                    self.observations = pt.empty(
                        (
                            o0.size(0),
                            time_length + 1,
                            conversion_object.observation_dimension,
                        )
                    )
                    self.observations[:, 0, :] = o0
                else:
                    self.observations[:, t, :] = conversion_object._get_observation(t)

                if state_availiable:
                    self.state[:, t, :] = conversion_object.x_t.squeeze(1)

    def __copy__(self):
        """
        返回一个新的 Observation_Queue
        具有相同的观测值和状态，设置在时间 0
        """
        try:
            out = Observation_Queue(
                xs=self.state, ys=self.observations, device=self.device
            )
        except AttributeError:
            out = Observation_Queue(ys=self.observations, device=self.device)
        return out

    def _get_observation(self, t):
        """获取时间 t 的观测值"""
        return self.observations[:, t, :]

    def save(
        self, path: str, start_idx: int, prefix: str = "", clear_folder=True
    ) -> None:
        """
        保存观测队列到文件
        
        参数:
        ----------
        path: str
            保存路径
        start_idx: int
            起始索引
        prefix: str
            文件名前缀
        clear_folder: bool
            是否清空文件夹
        """
        if clear_folder:
            if os.path.exists(path):
                print(f"Warning: This will overwrite the directory at path {path}")
                response = input("Input Y to confirm you want to do this:")
                if response != "Y" and response != "y":
                    print("Halting")
                    return
                try:
                    shutil.rmtree(path)
                except:
                    os.remove(path)
            os.mkdir(path)

        # 保存每个序列的观测值和状态
        for i in range(len(self.observations)):
            pt.save(
                self.observations[i].clone(),
                f"{path}/{prefix}_obs_{start_idx + i}_0.pt",
            )
            try:
                pt.save(
                    self.state[i].clone(), f"{path}/{prefix}_state_{start_idx + i}_0.pt"
                )
            except AttributeError:
                pass


# =============================================================================
# 第七部分：状态空间数据集
# 用于 PyTorch 数据加载的自定义数据集
# =============================================================================

class State_Space_Dataset(pt.utils.data.Dataset):
    """
    状态空间数据的自定义映射风格数据集
    适用于数据存储在单个目录中的情况
    
    允许状态或观测数据的不同维度具有不同的数据类型
    但使用前必须转换为通用类型
    
    参数:
    ----------
    path: str
        存储文件的目录路径
    
    prefix: str, 默认: ''
        所有文件的前缀
    
    lazy: bool, 默认: True
        如果为 True 则仅在需要时加载文件
        如果为 False 则在对象创建时加载所有数据
    
    files_per_obs: int, 默认: 1
        每个轨迹的观测值存储的文件数
    
    files_per_state: int, 默认: 1
        每个轨迹的状态存储的文件数
    
    obs_data_type: pt.dtype, 默认: None
        如果不为 None，所有观测数据将转换为给定类型
    
    state_data_type: pt.dtype, 默认: None
        如果不为 None，所有状态数据将转换为给定类型
    
    device: str 或 pt.device, 默认: 'cuda'
        放置所有张量的设备
    
    注意:
    ----------
    所有文件应该是 2D PyTorch 张量，使用 pt.save() 保存
    文件名可以以任意但不变的前缀开头
    观测值应标记为 'obs'，状态标记为 'state'
    文件被索引以链接来自同一轨迹的所有张量
    第二个索引表示要连接的张量的排序
    例如: 'directory/prefix_obs_1_1.pt'
    """

    def __init__(
        self,
        path: str,
        prefix: str = "",
        lazy: bool = True,
        files_per_obs: int = 1,
        files_per_state: int = 1,
        obs_data_type: pt.dtype = None,
        state_data_type: pt.dtype = None,
        device: str = "cuda",
        num_workers: int = 0,
    ) -> None:
        """
        初始化数据集
        
        参数:
        ----------
        path: str
            数据目录路径
        prefix: str
            文件前缀
        lazy: bool
            是否延迟加载
        files_per_obs: int
            每个观测值的文件数
        files_per_state: int
            每个状态的文件数
        obs_data_type: dtype
            观测数据类型
        state_data_type: dtype
            状态数据类型
        device: str
            运行设备
        num_workers: int
            数据加载的工作进程数
        """
        self.lazy = lazy
        self.device = device
        # 计算数据集长度
        self.length = (
            len([f for f in os.listdir(path) if f.startswith(f"{prefix}_obs")])
            // files_per_obs
        )
        self.workers = num_workers

        # 延迟加载模式：只保存参数
        if self.lazy:
            self.files_per_obs = files_per_obs
            self.files_per_state = files_per_state
            self.obs_data_type = obs_data_type
            self.state_data_type = state_data_type
            self.prefix = prefix
            self.dir = path
            return

        # 立即加载模式：加载所有数据
        try:
            self.data = [
                Observation_Queue(
                    xs=pt.concat(
                        tuple(
                            pt.load(f"{path}/{prefix}_state_{trajectory}_{i}.pt").to(
                                device=device, dtype=state_data_type
                            )
                            for i in range(files_per_state)
                        ),
                        dim=-1,
                    ),
                    ys=pt.concat(
                        tuple(
                            pt.load(f"{path}/{prefix}_obs_{trajectory}_{i}.pt").to(
                                device=device, dtype=obs_data_type
                            )
                            for i in range(files_per_obs)
                        ),
                        dim=-1,
                    ),
                )
                for trajectory in range(self.length)
            ]
        except FileNotFoundError:
            raise FileNotFoundError(
                "Tensor not found, make sure tensors use the approved naming scheme"
            )

    def __len__(self):
        """返回数据集长度"""
        return self.length

    def __getitem__(self, idx: int) -> Observation_Queue:
        """
        获取第 idx 个样本
        
        延迟加载模式下会动态读取文件
        """
        if self.lazy:
            try:
                return Observation_Queue(
                    xs=pt.concat(
                        tuple(
                            pt.load(f"{self.dir}/{self.prefix}_state_{idx}_{i}.pt").to(
                                device=self.device, dtype=self.state_data_type
                            )
                            for i in range(self.files_per_state)
                        ),
                        dim=-1,
                    ),
                    ys=pt.concat(
                        tuple(
                            pt.load(f"{self.dir}/{self.prefix}_obs_{idx}_{i}.pt").to(
                                device=self.device, dtype=self.obs_data_type
                            )
                            for i in range(self.files_per_obs)
                        ),
                        dim=-1,
                    ),
                )
            except FileNotFoundError as e:
                print(e)
                raise FileNotFoundError(
                    "Tensor not found, make sure tensors use the approved naming scheme"
                )
        return self.data[idx]

    def collate(self, batch: Iterable[Observation_Queue]) -> Observation_Queue:
        """
        批处理函数
        
        将多个 Observation_Queue 合并为一个
        """
        x_batch = pt.utils.data.default_collate([b.state for b in batch]).to(
            device=self.device
        )
        y_batch = pt.utils.data.default_collate([b.observations for b in batch]).to(
            device=self.device
        )
        return Observation_Queue(x_batch, y_batch)


# =============================================================================
# 第八部分：动态状态空间数据集
# 作为 Simulated_Objects 包装器的数据集
# =============================================================================

class dynamic_SS_dataset(pt.utils.data.IterableDataset):
    """
    作为 Simulated_Objects 包装器的数据集
    
    参数:
    -------------
    Template: Simulated_Object
        要复制的模拟对象模板
    
    batch_size: int, 默认: 1
        数据应生成的批次大小
    """

    def __init__(self, template: Simulated_Object, batch_size=1, num_workers: int = 0):
        """
        初始化动态数据集
        
        参数:
        ----------
        template: Simulated_Object
            模拟对象模板
        batch_size: int
            批量大小
        num_workers: int
            工作进程数
        """
        self.template = copy.copy(template)
        self.template.batches = batch_size
        self.workers = num_workers

    def _generate(self) -> Simulated_Object:
        """生成器：无限复制模板"""
        while True:
            yield copy.copy(self.template)

    def __iter__(self) -> Iterator[Simulated_Object]:
        """返回迭代器"""
        return iter(self._generate())

    def collate(self, batch) -> Simulated_Object:
        """
        批处理函数
        
        注意：动态数据集应使用批次大小为 1 的数据加载器
        真正的批次大小应在数据集创建时指定
        """
        if len(batch) != 1:
            warn(
                "Use a dataloader of batch size 1 with the dynamic dataset, the true batch size should be specified at dataset creation"
            )
        return batch[0]
