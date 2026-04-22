# =============================================================================
# simulationRS.py - IMM 粒子滤波器实现
# 本文件实现交互多模型粒子滤波器（Interacting Multiple Model Particle Filter）
# 
# 文件结构：
# IMM_Particle_Filter 类 - 实现 IMM-PF 算法
#   - __init__: 初始化粒子滤波器
#   - initialise: 初始化粒子状态
#   - scale_grad: 梯度缩放（自定义 autograd 函数）
#   - advance_one: 单步推进（核心算法）
#   - forward: 运行完整滤波过程
# =============================================================================

from dpf_rs.model import (
    Feynman_Kac,
    Simulated_Object,
)
from dpf_rs.utils import normalise_log_quantity
import numpy as np
from typing import Any, Union, Iterable
from copy import copy
from matplotlib import pyplot as plt
import torch as pt
from dpf_rs.resampling import Resampler
from torch import nn
from warnings import warn
from dpf_rs.resampling import batched_reindex


# =============================================================================
# IMM 粒子滤波器类
# 实现交互多模型粒子滤波算法
# =============================================================================

class IMM_Particle_Filter(nn.Module):
    """
    交互多模型粒子滤波器（IMM Particle Filter）
    
    实现通用的 Feynman-Kac 模型粒子滤波，支持 Bootstrap、Guided 和 Auxiliary 形式
    
    算法说明：
    ----------
    IMM-PF 是交互多模型（IMM）与粒子滤波（PF）的结合：
    - 同时维护多个动态模型（如8个模型）
    - 每个模型分配一定数量的粒子
    - 根据模型概率进行粒子交互（模型切换）
    - 使用重采样处理粒子退化
    
    与标准 PF 的区别：
    - 标准 PF：单个动态模型，所有粒子共享同一模型
    - IMM-PF：多个动态模型，粒子按模型分组，支持模型间切换
    
    参数说明：
    ----------
    model: Feynman_Kac
        Feynman-Kac 模型，定义状态转移、观测似然和切换动力学
    
    n_particles: int
        粒子总数，将被平均分配给各个模型
        例如：2000个粒子，8个模型，每个模型250个粒子
    
    resampler: Resampler
        重采样器，用于处理粒子退化
        常用：Soft_Resampler_Systematic（软系统重采样）
    
    ESS_threshold: float or int
        有效样本量（ESS）阈值，低于此值时进行重采样
        - 0或更低：从不重采样
        - n_particles或更高：每步都重采样
    
    device: str
        运行设备（'cuda' 或 'cpu'）
    
    IMMtype: str
        IMM 实现类型：
        - 'normal': 标准 IMM 实现
        - 'new': 论文提出的新实现（支持端到端训练）
        - 'OT': 使用最优传输重采样
    
    主要属性：
    ----------
    x_t: Tensor
        当前时刻的粒子状态，形状为 (batch, n_particles, state_dim)
    
    log_weights: Tensor
        粒子的对数权重，形状为 (batch, n_particles)
    
    log_normalised_weights: Tensor
        归一化的对数权重
    
    particles_per_model: int
        每个模型的粒子数 = n_particles // n_models
    
    使用示例：
    ----------
    >>> model = IMMPF(...)  # 创建 IMM 模型
    >>> pf = IMM_Particle_Filter(model, 2000, resampler, 2001, 'cuda', 'normal')
    >>> pf(sim_object, 50, [reporter])  # 运行50个时间步
    """

    def __init__(
        self,
        model: Feynman_Kac,
        n_particles: int,
        resampler: Resampler,
        ESS_threshold: Union[int, float], 
        device: str = 'cuda',
        IMMtype: str = 'normal'
    ) -> None:
        """
        初始化 IMM 粒子滤波器
        
        参数说明：
        ----------
        model: Feynman_Kac
            Feynman-Kac 模型
        n_particles: int
            粒子总数
        resampler: Resampler
            重采样器
        ESS_threshold: float or int
            ESS 阈值
        device: str
            运行设备
        IMMtype: str
            IMM 类型（'normal', 'new', 'OT'）
        """
        super().__init__()
        self.device = device
        self.resampler = resampler
        resampler.to(device=device)  # 将重采样器移到指定设备
        self.ESS_threshold = ESS_threshold
        self.n_particles = n_particles
        self.model = model
        
        # 检查滤波算法是否已设置
        if self.model.alg == self.model.PF_Type.Undefined:
            warn('Filtering algorithm not set')
        
        self.model.to(device=device)  # 将模型移到指定设备
        self.IMMtype = IMMtype

    def __copy__(self):
        """
        复制粒子滤波器
        
        创建当前粒子滤波器的浅拷贝
        """
        return IMM_Particle_Filter(
            copy(self.model),
            copy(self.truth),
            self.n_particles,
            self.resampler,
            self.ESS_threshold,
        )
    
    def initialise(self, truth: Simulated_Object) -> None:
        """
        初始化粒子状态
        
        在运行粒子滤波之前调用，初始化所有粒子的状态和权重
        
        初始化流程：
        1. 设置时间步 t = 0
        2. 获取模型数量 Nk
        3. 计算每个模型的粒子数
        4. 为每个模型采样初始粒子（使用 M_0_proposal）
        5. 计算初始权重（使用 log_f_t）
        6. 归一化权重
        
        参数说明：
        ----------
        truth: Simulated_Object
            模拟对象，包含真实状态和观测
        
        说明：
        - 粒子被平均分配给各个模型
        - 每个模型的粒子从该模型的初始分布中采样
        - 初始权重基于初始观测的似然
        """
        self.t = 0  # 初始化时间步
        self.truth = truth  # 保存模拟对象
        self.Nk = self.model.n_models  # 模型数量
        
        # 设置观测函数
        self.model.set_observations(self.truth._get_observation, 0)
        
        # 计算每个模型的粒子数
        # 使用整数除法，确保每个模型有相同数量的粒子
        self.particles_per_model = self.n_particles // self.Nk
        
        # 为每个模型采样初始粒子
        # M_0_proposal(k, batch_size, n_samples): 从模型k的初始分布采样
        # 结果拼接为 (batch, n_particles, state_dim)
        self.x_t = pt.concat([
            self.model.M_0_proposal(k, self.truth.state.size(0), self.particles_per_model) 
            for k in range(self.Nk)
        ], dim=1)
        
        # 计算初始对数权重
        # log_f_t(k, x, t): 计算模型k在时间t的观测对数似然
        self.log_weights = pt.concat([
            self.model.log_f_t(k, self.x_t[:, k*self.particles_per_model:(k+1)*self.particles_per_model, :], self.t) 
            for k in range(self.Nk)
        ], dim=1)
        
        # 归一化对数权重
        self.log_normalised_weights = normalise_log_quantity(self.log_weights)
        
        # 初始化粒子顺序（用于重采样跟踪）
        self.order = pt.arange(self.n_particles, device=self.device)
        
        # 标记已重采样
        self.resampled = True
        
        # 初始化重采样权重（均匀分布）
        self.resampled_weights = pt.zeros_like(self.log_weights) - np.log(self.n_particles)
        
        # 保存真实权重（用于后续计算）
        self.true_weights = self.log_normalised_weights

    class scale_grad(pt.autograd.Function):
        """
        梯度缩放自定义函数
        
        用于在反向传播时限制梯度大小，防止梯度爆炸
        
        前向传播：直接返回输入
        反向传播：将梯度裁剪到 [-10, 10] 范围
        
        使用场景：
        - 在 advance_one 中对 regime_probs 和 x_t 应用
        - 稳定训练过程，特别是在使用 'new' IMM 类型时
        """
        @staticmethod
        def forward(ctx: Any, input: pt.Tensor):
            """前向传播：直接返回输入的拷贝"""
            return input.clone()
        
        @staticmethod
        def backward(ctx, d_dinput):
            """反向传播：裁剪梯度到 [-10, 10]"""
            return pt.clip(d_dinput, -10, 10), None
    

    def advance_one(self) -> None:
        """
        单步推进粒子滤波器
        
        执行 IMM-PF 的核心算法（算法 10.3 的变体），推进一个时间步
        
        算法流程：
        1. 获取模型概率（切换概率）
        2. 调整模型概率（加入粒子权重）
        3. 计算总模型概率
        4. 计算重采样权重
        5. 对每个模型进行重采样
        6. 状态转移（提议分布）
        7. 更新权重（观测似然 + 切换概率）
        8. 归一化权重
        
        详细步骤说明：
        ----------------
        步骤1-3：模型概率计算
            regime_probs = model.get_regime_probs(x_t)
            - 获取从当前模型切换到其他模型的概率
            - 形状: (batch, n_particles, n_models)
            
            adj_regime_probs = regime_probs + true_weights[:,:,None]
            - 将粒子权重加入模型概率
            - 权重高的粒子对模型选择影响更大
        
        步骤4-6：重采样
            对每个模型 k：
                - 根据 regime_resampling_weights 重采样粒子
                - 从所有粒子中选择适合模型 k 的粒子
                - 返回新粒子 xs[k]、新权重 new_weights[k]、索引 indices[k]
        
        步骤7-8：状态转移和权重更新
            x_t[k] = M_t_proposal(k, xs[k], t)
            - 对重采样后的粒子应用模型 k 的状态转移
            
            log_weights = log_f_t(k, x_t[k], t) + new_weights[k] + tot_regime_probs
            - 新权重 = 观测似然 + 重采样权重 + 模型切换概率
        
        不同 IMM 类型的区别：
        --------------------
        'normal': 标准 IMM 实现
            - 使用 old_weights 计算权重修正
            - 适用于传统粒子滤波
        
        'new': 论文提出的新实现（支持端到端训练）
            - 使用 detach() 分离部分梯度
            - 添加额外的权重修正项
            - 支持反向传播训练
        
        'OT': 最优传输重采样
            - 使用最优传输进行重采样
            - 保持梯度的连续性
        """
        # 增加时间步
        self.t += 1
        
        # 步骤1：获取模型概率（切换概率）
        # regime_probs[k, i, j]: 粒子i从模型k切换到模型j的概率
        regime_probs = self.model.get_regime_probs(self.x_t)
        
        # 步骤2：调整模型概率（加入粒子权重）
        # 权重高的粒子对模型选择有更大影响
        adj_regime_probs = regime_probs + self.true_weights[:, :, None]
        
        # 应用梯度缩放，防止梯度爆炸
        adj_regime_probs = self.scale_grad.apply(adj_regime_probs)
        self.x_t = self.scale_grad.apply(self.x_t)
        
        # 步骤3：计算总模型概率（在粒子维度上求和）
        # tot_regime_probs[k]: 模型k的总概率
        tot_regime_probs = pt.logsumexp(adj_regime_probs, dim=1)
        
        # 步骤4：计算重采样权重
        # regime_resampling_weights[i, j, k]: 粒子i分配给模型k的权重
        regime_resampling_weights = adj_regime_probs - tot_regime_probs[:, None, :]
        
        # 初始化存储列表
        xs = [None] * self.Nk  # 新粒子
        indices = [None] * self.Nk  # 重采样索引
        new_weights = [None] * self.Nk  # 新权重
        
        # 保存旧粒子（用于后续计算）
        old_particles = self.x_t.clone()
        
        # 步骤5：对每个模型进行重采样
        for k in range(self.Nk):
            # 重采样：根据 regime_resampling_weights 选择粒子
            # 参数：particles_per_model（该模型的粒子数）、当前粒子、重采样权重
            # 返回：新粒子、新权重、选择的索引
            xs[k], new_weights[k], indices[k] = self.resampler(
                self.particles_per_model, 
                self.x_t.detach(), 
                regime_resampling_weights[:, :, k].detach()
            )
        
        # 拼接所有模型的新粒子
        self.x_t_1 = pt.concat(xs, dim=1)
        
        # 对于 'normal' 类型，计算旧权重（用于权重修正）
        if self.IMMtype == 'normal':
            indices = pt.concat(indices, dim=1)
            old_weights = batched_reindex(adj_regime_probs, indices)
        
        # 标记已重采样
        self.resampled = True
        self.resampled_weights = self.log_weights.clone()
        
        # 设置当前时刻的观测
        self.model.set_observations(self.truth._get_observation, self.t)
        
        # 步骤6：状态转移（提议分布）
        # 对每个模型应用状态转移：x_t = M_t_proposal(k, x_{t-1}, t)
        self.x_t = [self.model.M_t_proposal(k, xs[k], self.t) for k in range(self.Nk)]
        
        # 步骤7：更新权重
        # 根据 IMM 类型使用不同的权重更新公式
        if self.IMMtype == 'new':
            # 新实现：分离部分梯度，支持端到端训练
            self.log_weights = pt.concat([
                self.model.log_f_t(k, self.x_t[k], self.t) + 
                new_weights[k] + 
                tot_regime_probs[:, None, k].detach() 
                for k in range(self.Nk)
            ], dim=1)
        elif self.IMMtype == 'OT':
            # 最优传输：不分离梯度
            self.log_weights = pt.concat([
                self.model.log_f_t(k, self.x_t[k], self.t) + 
                new_weights[k] + 
                tot_regime_probs[:, None, k] 
                for k in range(self.Nk)
            ], dim=1)
        else:
            # 标准实现：使用 old_weights 进行权重修正
            self.log_weights = pt.concat([
                self.model.log_f_t(k, self.x_t[k], self.t) + 
                new_weights[k] + 
                old_weights[:, k*self.particles_per_model:(k+1)*self.particles_per_model, k] - 
                old_weights[:, k*self.particles_per_model:(k+1)*self.particles_per_model, k].detach() + 
                tot_regime_probs[:, None, k].detach() 
                for k in range(self.Nk)
            ], dim=1)
        
        # 拼接所有模型的粒子
        self.x_t = pt.concat(self.x_t, dim=1)
        
        # 步骤8：归一化权重
        self.true_weights = normalise_log_quantity(self.log_weights)
        self.log_normalised_weights = self.true_weights
        
        # 对于 'new' 类型且处于训练模式，添加额外的权重修正
        if self.IMMtype == 'new' and self.training:
            weights = [None] * self.Nk
            for k in range(self.Nk):
                start_range = k * self.particles_per_model
                end_range = (k + 1) * self.particles_per_model
                # 计算提议分布的权重修正
                weights[k] = adj_regime_probs[:, None, :, k] + self.model.log_M_t(
                    k, self.x_t[:, start_range:end_range], old_particles, self.t
                )
            weights = pt.concat(weights, dim=1)
            weights = pt.logsumexp(weights, dim=2)
            # 添加权重修正（分离梯度）
            self.log_weights = self.log_weights + weights - weights.detach()
            self.log_normalised_weights = normalise_log_quantity(self.log_weights)
            self.true_weights = self.log_normalised_weights


    def forward(self, sim_object: Simulated_Object, iterations: int, statistics: Iterable):
        """
        运行粒子滤波器
        
        执行完整的粒子滤波过程，收集统计信息
        
        参数说明：
        ----------
        sim_object: Simulated_Object
            模拟对象，包含真实状态和观测序列
        
        iterations: int
            运行的时间步数
        
        statistics: Iterable[Reporter]
            统计信息收集器列表
            每个 Reporter 在滤波过程中收集特定信息（如状态估计、权重等）
        
        返回：
        ----------
        statistics: Iterable[Reporter]
            包含收集结果的 Reporter 列表
        
        执行流程：
        ----------
        1. 初始化粒子状态
        2. 初始化所有统计收集器
        3. 循环 iterations+1 次：
           - 评估统计收集器（记录当前状态）
           - 如果达到 iterations，退出循环
           - 执行单步推进（advance_one）
        4. 最终化统计收集器
        5. 返回统计结果
        
        使用示例：
        ----------
        >>> from dpf_rs.results import Mean
        >>> reporter = Mean()
        >>> pf(sim_object, 50, [reporter])
        >>> mean_estimate = reporter.result  # 获取状态估计均值
        """
        # 初始化粒子滤波器
        self.initialise(sim_object)

        # 初始化所有统计收集器
        for stat in statistics:
            stat.initialise(self, iterations)

        # 运行粒子滤波循环
        for _ in range(iterations + 1):
            # 评估统计收集器（记录当前粒子状态）
            for stat in statistics:
                stat.evaluate(PF=self)
            
            # 如果达到指定时间步，退出循环
            if self.t == iterations:
                break
            
            # 执行单步推进
            self.advance_one()
        
        # 最终化统计收集器（计算最终结果）
        stat.finalise(self)

        return statistics
