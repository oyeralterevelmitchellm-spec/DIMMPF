# =============================================================================
# Net.py - 神经网络模型定义
# 本文件定义了各种切换模型、神经网络组件和粒子滤波模型
# 
# 文件结构：
# 第一部分：切换模型（Switching Dynamics）- 定义不同的模型切换机制
# 第二部分：神经网络组件 - 基础的神经网络模块
# 第三部分：粒子滤波模型 - 各种粒子滤波算法的实现
# 第四部分：基线模型 - LSTM和Transformer用于对比
# =============================================================================

from typing import Callable, List
import torch as pt
from dpf_rs.model import *
from numpy import sqrt
from dpf_rs.utils import batched_select


# =============================================================================
# 第一部分：切换模型（Switching Dynamics）
# 定义不同的模型切换机制
# =============================================================================

class Markov_Switching(pt.nn.Module):
    """
    马尔可夫切换模型
    
    使用转移概率矩阵控制模型之间的切换
    每个模型以一定概率保持或切换到其他模型
    
    参数说明：
    ----------
    n_models: int
        模型数量（如8个模型）
    switching_diag: float
        对角线元素（保持当前模型的概率，如0.8）
    switching_diag_1: float
        次对角线元素（切换到相邻模型的概率，如0.15）
    dyn: str
        动态类型（"Boot": Bootstrap（高斯噪声）, "Uni": 均匀分布, "Deter": 确定性）
    device: str
        运行设备（"cuda"或"cpu"）
    """

    def __init__(
        self,
        n_models: int,
        switching_diag: float,
        switching_diag_1: float,
        dyn="Boot",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.dyn = dyn
        self.n_models = n_models
        
        # 构建转移概率向量
        # 非对角线元素均匀分配剩余概率
        # 例如：8个模型，保持概率0.8，相邻切换概率0.15
        # 则其他6个模型的概率 = (1 - 0.8 - 0.15) / 6 = 0.0083
        tprobs = pt.ones(n_models) * (
            (1 - switching_diag - switching_diag_1) / (n_models - 2)
        )
        # tprobs = [0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083]
        tprobs[0] = switching_diag
        # tprobs = [0.8, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083]
        tprobs[1] = switching_diag_1
        # tprobs = [0.8, 0.15, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0083]

        self.switching_vec = pt.log(tprobs).to(device=device)
        # 将转移概率转换为对数概率并保存到指定设备
        # 原因：
        #   1. 数值稳定性：避免概率相乘时的下溢问题
        #   2. 计算效率：概率乘法转换为对数加法
        #   3. 标准做法：粒子滤波中普遍使用对数权重
        # 转换示例：
        #   tprobs = [0.8, 0.15, 0.0083, ...]
        #   pt.log(tprobs) = [-0.223, -1.897, -4.787, ...]
        # 后续使用：
        #   - forward 方法：执行模型切换采样
        #   - get_regime_probs 方法：计算切换概率矩阵

        self.dyn = dyn

    def init_state(self, batches, n_samples):
        """
        初始化模型状态
        
        参数说明：
        ----------
        batches: int
            批量大小
        n_samples: int
            每批样本数
        
        返回：
        ----------
        Tensor: 初始模型索引，形状为 (batches, n_samples, 1)
        
        说明：
        - "Uni": 均匀分布初始化，每个模型概率相等
        - "Deter": 确定性初始化，循环分配模型（0,1,2,...,7,0,1,2...）
        - 其他: 根据转移概率随机采样初始化
        """
        # 初始化模型状态概率
        # 根据 dyn 参数选择不同的初始化方式
        if self.dyn == "Uni":
            # 均匀分布初始化
            # 计算：pt.ones(n_models) / n_models
            # 示例（n_models=8）：[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
            # 特点：所有模型具有相同的初始概率（12.5%）
            # 使用场景：无先验知识，所有模型等可能
            self.probs = pt.ones(self.n_models) / self.n_models
        else:
            # 使用预定义的转移概率（非均匀分布）
            # 计算：pt.exp(switching_vec) 将对数概率转回概率
            # 示例：switching_vec = [-0.223, -1.897, -4.787, ...]
            #       pt.exp(...) = [0.8, 0.15, 0.0083, ...]
            # 特点：模型0有80%概率被选中，模型1有15%，其他模型约0.83%
            # 使用场景：有先验知识，按马尔可夫转移概率初始化
            self.probs = pt.exp(self.switching_vec)
        
        if self.dyn == "Deter":
            # 确定性初始化：循环分配模型（无随机性）
            # 构建过程：
            #   1. pt.arange(n_models): 创建序列 [0, 1, 2, ..., 7]
            #   2. .tile((batches, n_samples // n_models)): 重复平铺
            #      示例：batches=2, n_samples=96, n_models=8
            #      每行：[0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7, ...] 重复12次
            #   3. .unsqueeze(2): 增加维度，形状从 (2, 96) 变为 (2, 96, 1)
            #      目的：符合粒子状态的标准形状 (batches, n_particles, state_dim)
            # 特点：
            #   - 无随机性：每次运行结果相同
            #   - 均匀分配：每个模型分配相同数量的粒子
            #   - 可重复：适用于调试和对比实验
            return (
                pt.arange(self.n_models, device=self.device)
                .tile((batches, n_samples // self.n_models))
                .unsqueeze(2)
            )
        
        # 随机采样初始化（当 dyn 不是 "Deter" 时执行）
        # 构建过程：
        #   1. pt.multinomial(probs, batches * n_samples, True): 多项式采样
        # pt.multinomial(...) - 按概率随机采样 
        # 从 0 到 n_models-1 中随机选择索引，每个数字被选中的概率相等（均匀分布）
        # 返回形如 [3, 7, 1, 0, 4, 2, 5, 6, ...] 的张量
        #      - probs: 概率分布（这里是均匀分布 [1,1,1,1,1,1,1,1]）
        #      - batches * n_samples: 采样次数（总样本数，如 2*100=200）
        #      - True: 允许重复采样
        #      示例：返回 [3, 7, 1, 0, 4, 2, 5, 6, 3, 1, ...] 共200个随机索引
        #      每个数字在 0-7 之间，按概率分布随机选择
        #   2. .reshape((batches, n_samples, 1)): 重塑为3D张量
        #      形状变化：(200,) → (2, 100, 1)
        #      符合粒子状态的标准形状 (batches, n_particles, state_dim)
        #   3. .to(device=self.device): 移动到指定设备（cuda/cpu）
        # 特点：
        #   - 有随机性：每次运行结果不同
        #   - 可按概率分布：若使用非均匀概率，则按该分布采样
        #   - 使用场景：实际训练、测试
        # 与确定性初始化的对比：
        #   - 确定性：无随机性，顺序循环，结果可重复
        #   - 随机采样：有随机性，按概率随机，结果不同
        return (
            pt.multinomial(pt.ones(self.n_models), batches * n_samples, True)
            .reshape((batches, n_samples, 1))
            .to(device=self.device)
        )

    def forward(self, x_t_1, t):
        """
        前向传播：执行模型切换
        
        参数说明：
        ----------
        x_t_1: Tensor
            前一时刻的状态，形状为 (batches, n_samples, state_dim)
        t: int
            当前时间步
        
        返回：
        ----------
        Tensor: 新的模型索引
        
        说明：
        - "Deter": 确定性切换，循环分配模型
        - 其他: 根据转移概率随机采样，计算新模型索引
        """
        if self.dyn == "Deter":
            # 确定性切换
            return (
                pt.arange(self.n_models, device=self.device)
                .tile((x_t_1.size(0), x_t_1.size(1) // self.n_models))
                .unsqueeze(2)
            )
        
        # 随机切换：根据转移概率采样
        shifts = (
            pt.multinomial(self.probs, x_t_1.size(0) * x_t_1.size(1), True)
            .to(self.device)
            .reshape([x_t_1.size(0), x_t_1.size(1)])
        )
        new_models = pt.remainder(shifts + x_t_1[:, :, 0], self.n_models)
        return new_models.unsqueeze(2)

    def get_log_probs(self, x_t, x_t_1):
        """
        计算切换的对数概率
        
        用于计算粒子权重，根据模型切换计算对数概率
        """
        shifts = x_t[:, :, 0] - x_t_1[:, :, 0]
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return self.switching_vec[shifts]

    def get_regime_probs(self, x_t_1):
        """
        获取模型概率分布
        
        用于IMM算法中的模型概率混合，计算每个模型的对数概率
        """
        ks = pt.arange(0, self.n_models, device=self.device)
        shifts = ks[None, None, :] - x_t_1[:, :, 0:1]
        shifts = pt.remainder(shifts, self.n_models).to(int)
        return self.switching_vec[shifts].reshape(shifts.size())

    def R_0(self, batches, n_samples, k):
        """
        初始化辅助变量（用于IMM）
        
        参数说明：
        ----------
        batches: int
            批量大小
        n_samples: int
            每批样本数
        k: int
            模型索引
        
        返回：
        ----------
        Tensor: 初始化的辅助变量，值为k
        """
        return pt.ones((batches, n_samples, 1), device=self.device) * k

    def R_t(self, r_t_1, k):
        """
        更新辅助变量（用于IMM）
        
        参数说明：
        ----------
        r_t_1: Tensor
            前一时刻的辅助变量
        k: int
            模型索引
        
        返回：
        ----------
        Tensor: 更新后的辅助变量，值为k
        """
        return pt.ones_like(r_t_1) * k


class Polya_Switching(pt.nn.Module):
    """
    波利亚切换模型
    
    基于狄利克雷过程的中国餐馆过程（Chinese Restaurant Process）
    使用计数器跟踪模型选择历史，倾向于选择已频繁选择的模型（富者愈富效应）
    
    参数说明：
    ----------
    n_models: int
        模型数量
    dyn: str
        动态类型（"Boot", "Uni", "Deter"）
    device: str
        运行设备
    """

    def __init__(self, n_models, dyn, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.dyn = dyn
        self.n_models = n_models
        self.ones_vec = pt.ones(n_models)

    def init_state(self, batches, n_samples):
        """
        初始化状态，包含模型索引和计数器
        
        返回：
        ----------
        Tensor: 初始状态，包含模型索引和计数器
            - 第0维：模型索引
            - 第1到n_models维：每个模型的选择次数计数器
        """
        self.scatter_v = pt.zeros(
            (batches, n_samples, self.n_models), device=self.device
        )
        i_models = (
            pt.multinomial(self.ones_vec, batches * n_samples, True)
            .reshape((batches, n_samples, 1))
            .to(device=self.device)
        )
        return pt.concat(
            (
                i_models,
                pt.ones((batches, n_samples, self.n_models), device=self.device),
            ),
            dim=2,
        )

    def forward(self, x_t_1, t):
        """
        前向传播：更新计数器并选择新模型
        
        说明：
        - 更新计数器：增加当前模型的计数
        - 选择新模型：
          - "Uni": 均匀采样（不考虑历史）
          - 其他: 根据计数器采样（倾向于选择已频繁选择的模型）
        """
        self.scatter_v.zero_()
        self.scatter_v.scatter_(2, x_t_1[:, :, 0].unsqueeze(2).to(int), 1)
        c = x_t_1[:, :, 1:] + self.scatter_v
        
        if self.dyn == "Uni":
            # 均匀采样
            return pt.concat(
                (
                    pt.multinomial(self.ones_vec, x_t_1.size(0) * x_t_1.size(1), True)
                    .to(self.device)
                    .reshape([x_t_1.size(0), x_t_1.size(1), 1]),
                    c,
                ),
                dim=2,
            )
        
        # 根据计数器采样（倾向于选择已频繁选择的模型）
        return pt.concat(
            (
                pt.multinomial(c.reshape(-1, self.n_models), 1, True)
                .to(self.device)
                .reshape([x_t_1.size(0), x_t_1.size(1), 1]),
                c,
            ),
            dim=2,
        )

    def get_log_probs(self, x_t, x_t_1):
        """
        计算对数概率
        
        根据计数器计算模型选择的概率
        """
        probs = x_t[:, :, 1:]
        probs /= pt.sum(probs, dim=2, keepdim=True)
        s_probs = batched_select(probs, x_t_1[:, :, 1].to(int))
        return pt.log(s_probs)

    def get_regime_probs(self, x_t_1):
        """
        获取模型概率
        
        计算每个模型的概率（归一化计数器）
        """
        probs = x_t_1 / pt.sum(x_t_1, dim=2, keepdim=True)
        return pt.log(probs)

    def R_0(self, batches, n_samples, k):
        """
        初始化辅助变量
        
        参数说明：
        ----------
        batches: int
            批量大小
        n_samples: int
            每批样本数
        k: int
            模型索引
        
        返回：
        ----------
        Tensor: 初始化的辅助变量，模型k的计数为2，其他为1
        """
        t = pt.ones((batches, n_samples, self.n_models), device=self.device)
        t[:, :, k] = 2
        return t

    def R_t(self, r_t_1, k):
        """
        更新辅助变量
        
        增加模型k的计数
        """
        temp = r_t_1
        temp[:, :, k] = temp[:, :, k] + 1
        return temp


class Erlang_Switching(pt.nn.Module):
    """
    爱尔朗切换模型
    
    基于爱尔朗分布的切换机制
    模型在切换到新模型前会在当前模型停留一段时间（持续时间建模）
    
    参数说明：
    ----------
    n_models: int
        模型数量
    dyn: str
        动态类型
    device: str
        运行设备
    """

    def __init__(self, n_models, dyn, device: str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.dyn = dyn
        self.n_models = n_models
        self.ones_vec = pt.ones(n_models)
        # 定义前后索引用于相邻模型切换
        self.permute_backward = pt.remainder(
            pt.arange(self.n_models) + 1, self.n_models
        )
        self.permute_forward = pt.remainder(pt.arange(self.n_models) - 1, self.n_models)

    def init_state(self, batches, n_samples):
        """
        初始化状态
        
        返回：
        ----------
        Tensor: 初始状态，包含模型索引和计数器
        """
        self.scatter_v = pt.zeros(
            (batches, n_samples, self.n_models), device=self.device
        )
        i_models = (
            pt.multinomial(self.ones_vec, batches * n_samples, True)
            .reshape((batches, n_samples, 1))
            .to(device=self.device)
        )
        return pt.concat(
            (
                i_models,
                pt.zeros((batches, n_samples, self.n_models + 1), device=self.device),
            ),
            dim=2,
        )

    def forward(self, x_t_1, t):
        """
        前向传播：执行爱尔朗切换
        
        说明：
        - 使用计数器跟踪在当前模型的停留时间
        - 小概率随机探索（0.01）
        - 大概率根据停留时间决定是否切换
        - 切换到相邻模型的概率不同（前向0.6，后向0.4）
        """
        tensor_shape = (x_t_1.size(0), x_t_1.size(1))
        self.scatter_v.zero_()
        self.scatter_v.scatter_(2, x_t_1[:, :, 0].unsqueeze(2).to(int), 1)
        
        # 基础概率（小概率随机探索）
        self.true_probs = (
            pt.ones(self.n_models, device=self.device) * (0.01 / self.n_models)
        ).reshape((1, 1, -1))
        
        output = x_t_1[:, :, 1:].clone()
        mask = self.scatter_v.to(dtype=bool)
        counts = output[:, :, :-1][mask].reshape(tensor_shape).unsqueeze(2)

        # 计算停留和切换概率
        stay_probs = self.scatter_v
        change_probs = (
            self.scatter_v[:, :, self.permute_forward] * 0.6
            + self.scatter_v[:, :, self.permute_backward] * 0.4
        )
        
        # 混合概率
        mixes = (pt.rand(tensor_shape, device=self.device) > 0.01).unsqueeze(2)
        draw_probs = (
            pt.ones(self.n_models, device=self.device) / self.n_models
        ).reshape((1, 1, -1))

        target_counts = output[:, :, -1].unsqueeze(2)
        self.true_probs = self.true_probs + pt.where(
            counts == target_counts, change_probs * 0.2 + stay_probs * 0.8, stay_probs
        ) * (1 - 0.01)

        # 更新计数器
        subtract = (pt.rand(tensor_shape, device=self.device) < 0.2).unsqueeze(2)
        fake_output = output.clone()
        fake_output[:, :, -1] = fake_output[:, :, -1] + 1
        output = pt.where(subtract, fake_output, output)

        fake_output = output.clone()
        fake_output[:, :, -1] = 0
        fake_output[:, :, :-1] = fake_output[:, :, :-1] + self.scatter_v

        target_counts = output[:, :, -1].unsqueeze(2)
        output = pt.where(
            pt.logical_or(counts < target_counts, pt.logical_not(mixes)),
            fake_output,
            output,
        )
        
        draw_probs = pt.where(
            pt.logical_and(counts < target_counts, mixes), change_probs, draw_probs
        )
        draw_probs = pt.where(
            pt.logical_and(counts >= target_counts, mixes), stay_probs, draw_probs
        )
        
        if self.dyn == "Uni":
            return pt.concat(
                (
                    pt.multinomial(self.ones_vec, x_t_1.size(0) * x_t_1.size(1), True)
                    .to(self.device)
                    .reshape([x_t_1.size(0), x_t_1.size(1), 1]),
                    output,
                ),
                dim=2,
            )
        
        return pt.concat(
            (
                pt.multinomial(draw_probs.reshape(-1, self.n_models), 1, True)
                .to(self.device)
                .reshape([x_t_1.size(0), x_t_1.size(1), 1]),
                output,
            ),
            dim=2,
        )

    def get_log_probs(self, x_t, x_t_1):
        """
        获取对数概率
        
        返回当前的真实概率
        """
        return self.true_probs

    def get_regime_probs(self, x_t_1):
        """
        获取模型概率
        
        根据当前状态计算每个模型的概率
        """
        output = pt.ones(
            (x_t_1.size(0), x_t_1.size(1), self.n_models), device=self.device
        ) * (0.01 / self.n_models)
        scatter_k = pt.zeros_like(output)
        scatter_k.scatter_(2, x_t_1[:, :, -1].unsqueeze(-1).to(int), 1)
        change_probs = (
            scatter_k[:, :, self.permute_forward] * 0.6
            + scatter_k[:, :, self.permute_backward] * 0.4
        )
        output = (
            pt.where(
                x_t_1[:, :, -2:-1] == 0, change_probs * 0.2 + scatter_k * 0.8, scatter_k
            )
            * 0.99
            + output
        )
        return pt.log(output)

    def R_0(self, batches, n_samples, k):
        """
        初始化辅助变量
        
        参数说明：
        ----------
        batches: int
            批量大小
        n_samples: int
            每批样本数
        k: int
            模型索引
        
        返回：
        ----------
        Tensor: 初始化的辅助变量，最后一维为k
        """
        t = pt.zeros((batches, n_samples, self.n_models + 6), device=self.device)
        t[:, :, -1] = k
        return t

    def R_t(self, r_t_1, k):
        """
        更新辅助变量
        
        更新计数器和当前模型索引
        """
        has_changed = r_t_1.clone()
        view_tensor = batched_select(has_changed, r_t_1[:, :, -1])
        view_tensor[:, :] = view_tensor + 1
        has_changed[:, :, -2] = r_t_1[:, :, k]
        has_changed[:, :, -1] = k
        
        from_mixing = pt.rand(
            (r_t_1.size(0), r_t_1.size(1), 1), device=self.device
        ) < 1 / (99 * self.n_models + 1)
        mixing_cond = pt.logical_or(k != r_t_1[:, :, -1, None], from_mixing)
        output = pt.where(mixing_cond, has_changed, r_t_1)

        from_decrease = (
            pt.rand((r_t_1.size(0), r_t_1.size(1), 1), device=self.device) > 0.2
        )
        decrease = r_t_1.clone()
        decrease[:, :, -2] = decrease[:, :, -2] - 1
        output = pt.where(
            pt.logical_or(
                mixing_cond, pt.logical_or(from_decrease, r_t_1[:, :, -2:-1] == 0)
            ),
            output,
            decrease,
        )
        return output


class NN_Switching(pt.nn.Module):
    """
    神经网络切换模型
    
    使用循环神经网络（RNN）学习切换动力学
    这是RLPF和DIMMPF中使用的可学习切换模型
    
    参数说明：
    ----------
    n_models: int
        模型数量
    recurrent_length: int
        循环层维度（RNN隐藏层大小）
    dyn: str
        动态类型
    device: str
        运行设备
    softness: float
        软化参数（用于重要性采样校正）
    """

    def __init__(self, n_models, recurrent_length, dyn, device, softness):
        super().__init__()
        self.device = device
        self.r_length = recurrent_length
        self.n_models = n_models
        
        # 定义神经网络层
        # 遗忘门：决定保留多少历史信息（类似LSTM的遗忘门）
        self.forget = pt.nn.Sequential(
            pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid()
        )
        # 自遗忘门：循环自连接（对历史信息的自调节）
        self.self_forget = pt.nn.Sequential(
            pt.nn.Linear(recurrent_length, recurrent_length), pt.nn.Sigmoid()
        )
        # 缩放门：对输入进行缩放
        self.scale = pt.nn.Sequential(
            pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid()
        )
        # 输入转换：将输入映射到循环层
        self.to_reccurrent = pt.nn.Sequential(
            pt.nn.Linear(n_models, recurrent_length), pt.nn.Tanh()
        )
        # 输出层：从循环层输出模型概率
        self.output_layer = pt.nn.Sequential(
            pt.nn.Linear(recurrent_length, recurrent_length),
            pt.nn.Tanh(),
            pt.nn.Linear(recurrent_length, n_models),
        )
        self.dyn = dyn
        self.softness = softness

    def init_state(self, batches, n_samples):
        """
        初始化状态
        
        返回：
        ----------
        Tensor: 初始状态，包含模型索引和循环层状态
        """
        self.probs = pt.ones(self.n_models) / self.n_models
        self.true_probs = pt.ones(self.n_models) / self.n_models
        i_models = (
            pt.multinomial(self.probs, batches * n_samples, True)
            .reshape((batches, n_samples, 1))
            .to(device=self.device)
        )
        if self.r_length > 0:
            return pt.concat(
                (
                    i_models,
                    pt.zeros((batches, n_samples, self.r_length), device=self.device),
                ),
                dim=2,
            )
        else:
            return i_models

    def forward(self, x_t_1, t):
        """
        前向传播：RNN单元
        
        说明：
        - 使用one-hot编码当前模型
        - 通过遗忘门和输入转换更新循环状态
        - 输出模型概率分布
        - 使用软化参数进行重要性采样校正
        """
        old_model = x_t_1[:, :, 0].to(int).unsqueeze(2)
        
        # One-hot编码
        one_hot = pt.zeros(
            (old_model.size(0), old_model.size(1), self.n_models), device=self.device
        )
        one_hot = pt.scatter(one_hot, 2, old_model, 1)
        
        # 获取历史信息
        old_recurrent = x_t_1[:, :, 1:]
        
        # RNN更新
        c = old_recurrent * self.self_forget(old_recurrent)
        c *= self.forget(one_hot)
        c += self.to_reccurrent(one_hot)
        
        # 计算输出概率
        probs = pt.abs(self.output_layer(c))
        self.true_probs = probs / pt.sum(probs, dim=2, keepdim=True)
        self.correction = (
            self.softness * self.true_probs.detach()
            + (1 - self.softness) / self.n_models
        )
        probs = self.correction
        
        return pt.concat(
            (
                pt.multinomial(probs.reshape(-1, self.n_models), 1, True)
                .to(self.device)
                .reshape([x_t_1.size(0), x_t_1.size(1), 1]),
                c,
            ),
            dim=2,
        )

    def get_weight(self, x_t, x_t_1):
        """
        计算重要性权重（用于粒子滤波）
        
        说明：
        - 计算真实概率与提议概率的比值
        - 用于粒子滤波中的重要性采样校正
        """
        models = x_t[:, :, 0].to(int)
        probs = batched_select(
            self.true_probs.reshape(-1, self.n_models), models.flatten()
        ).reshape(x_t.size(0), x_t.size(1))
        corrections = batched_select(
            self.correction.reshape(-1, self.n_models), models.flatten()
        ).reshape(x_t.size(0), x_t.size(1))
        return pt.log(probs / corrections + 1e-7)

    def get_regime_probs(self, r_t_1):
        """
        获取模型概率
        
        根据循环状态计算每个模型的概率
        """
        probs = pt.abs(self.output_layer(r_t_1) + 1e-7)
        probs = probs / pt.sum(probs, dim=2, keepdim=True)
        return pt.log(probs)

    def R_0(self, batches, n_samples, k):
        """
        初始化辅助变量
        
        参数说明：
        ----------
        batches: int
            批量大小
        n_samples: int
            每批样本数
        k: int
            模型索引
        
        返回：
        ----------
        Tensor: 初始化的辅助变量
        """
        return self.R_t(
            pt.zeros((batches, n_samples, self.r_length), device=self.device), k
        )

    def R_t(self, r_t_1, k):
        """
        更新辅助变量
        
        根据模型k更新循环状态
        """
        self.zero_vec = pt.zeros(self.n_models, device=self.device)
        self.zero_vec[k] = 1
        c = r_t_1 * self.self_forget(r_t_1)
        c = c * self.forget(self.zero_vec)
        c = c + self.to_reccurrent(self.zero_vec)
        return c


# =============================================================================
# 第二部分：神经网络组件
# 基础的神经网络模块
# =============================================================================

class Recurrent_Unit(pt.nn.Module):
    """
    循环单元
    
    自定义的RNN单元，结合了LSTM和GRU的特点
    使用两个门控机制控制信息流
    
    参数说明：
    ----------
    input: int
        输入维度
    hidden: int
        隐藏层维度
    output: int
        输出维度
    out_layers: int
        输出层数
    """

    def __init__(self, input, hidden, output, out_layers):
        super().__init__()
        self.tanh = pt.nn.Tanh()
        self.sigmoid = pt.nn.Sigmoid()
        # 遗忘门：控制保留多少历史信息
        self.forget = pt.nn.Linear(input, hidden)
        # 输入转换：将输入映射到隐藏层
        self.to_hidden = pt.nn.Linear(input, hidden)
        # 温度门：控制输入的影响程度
        self.temper = pt.nn.Linear(input, hidden)
        # 输出网络
        self.out = Simple_NN(input + hidden, hidden, output, out_layers)

    def forward(self, in_vec, hidden_vec):
        """
        前向传播
        
        说明：
        - a = hidden * sigmoid(forget(in))：保留的历史信息
        - b = sigmoid(temper(in)) * tanh(to_hidden(in))：新的输入信息
        - hidden_out = a + b：更新后的隐藏状态
        - out = net(concat(in, hidden_out))：输出
        """
        a = hidden_vec * self.sigmoid(self.forget(in_vec))
        b = self.sigmoid(self.temper(in_vec)) * self.tanh(self.to_hidden(in_vec))
        hidden_out = a + b
        out = self.out(pt.concat((in_vec, hidden_out), dim=-1))
        return pt.concat((out, hidden_out), dim=-1)


class Likelihood_NN(pt.nn.Module):
    """
    似然神经网络
    
    用于计算观测似然
    输入观测和状态，输出似然值
    
    参数说明：
    ----------
    input: int
        输入维度
    hidden: int
        隐藏层维度
    output: int
        输出维度
    """

    def __init__(self, input, hidden, output):
        super().__init__()
        self.net = pt.nn.Sequential(
            pt.nn.Linear(input, hidden),
            pt.nn.Tanh(),
            pt.nn.Linear(hidden, hidden),
            pt.nn.Tanh(),
            pt.nn.Linear(hidden, output),
        )

    def forward(self, in_vec):
        """
        前向传播
        
        输入形状：(batches, n_samples, input_dim)
        输出形状：(batches, n_samples, output_dim)
        """
        return self.net(in_vec.unsqueeze(1)).squeeze()


class Simple_NN(pt.nn.Module):
    """
    简单神经网络
    
    基础的全连接网络，用于各种组件
    
    参数说明：
    ----------
    input: int
        输入维度
    hidden: int
        隐藏层维度
    output: int
        输出维度
    layers: int
        层数
    """

    def __init__(self, input, hidden, output, layers):
        super().__init__()
        nn_layers = [pt.nn.Linear(input, hidden), pt.nn.Tanh()]
        for i in range(layers - 2):
            nn_layers += [pt.nn.Linear(hidden, hidden), pt.nn.Tanh()]
        nn_layers += [pt.nn.Linear(hidden, output)]
        self.net = pt.nn.Sequential(*tuple(nn_layers))

    def forward(self, in_vec):
        """
        前向传播
        
        输入形状：(batches, n_samples, input_dim)
        输出形状：(batches, n_samples, output_dim)
        """
        return self.net(in_vec)


# =============================================================================
# 第三部分：粒子滤波模型
# 各种粒子滤波算法的实现
# =============================================================================

class PF(SSM):
    """
    基础粒子滤波模型（Particle Filter）
    
    用于生成模拟数据的标准粒子滤波
    状态转移方程：x_t = a[k] * x_{t-1} + b[k] + noise
    观测方程：y_t = a[k] * sqrt(|x_t|) + b[k] + noise
    
    参数说明：
    ----------
    a: List[int]
        状态转移系数列表（每个模型一个）
    b: List[int]
        状态转移偏置列表（每个模型一个）
    var_s: float
        噪声方差
    switching_dyn: Module
        切换动力学模型
    dyn: str
        动态类型（"Boot"或"Guided"）
    device: str
        运行设备
    """

    def set_observations(self, get_observation: Callable, t: int):
        """
        设置观测值
        
        参数说明：
        ----------
        get_observation: Callable
            获取观测值的函数
        t: int
            当前时间步
        """
        self.y = self.reindexed_array(
            t - 1, [get_observation(t - 1), get_observation(t)]
        )

    def __init__(
        self,
        a: List[int],
        b: List[int],
        var_s: float,
        switching_dyn: pt.nn.Module,
        dyn="Boot",
        device: str = "cuda",
    ):
        super().__init__(device)
        self.n_models = len(a)
        self.a = pt.tensor(a, device=device)
        self.b = pt.tensor(b, device=device)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.var_factor = -1 / (2 * var_s)
        self.y_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        
        if dyn == "Boot":
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def M_0_proposal(self, batches: int, n_samples: int):
        """
        时间0的提议分布
        
        参数说明：
        ----------
        batches: int
            批量大小
        n_samples: int
            每批样本数
        
        返回：
        ----------
        Tensor: 初始状态，包含位置和模型索引
        """
        init_locs = (
            self.init_x_dist.sample([batches, n_samples])
            .to(device=self.device)
            .unsqueeze(2)
        )
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        return pt.cat((init_locs, init_regimes), dim=2)

    def M_t_proposal(self, x_t_1, t: int):
        """
        时间t的提议分布
        
        参数说明：
        ----------
        x_t_1: Tensor
            前一时刻的状态
        t: int
            当前时间步
        
        返回：
        ----------
        Tensor: 新状态，包含位置和模型索引
        
        说明：
        - 根据切换模型选择新模型
        - 根据新模型索引选择对应的a和b
        - 计算新位置：x_t = a[k] * x_{t-1} + b[k] + noise
        """
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(
            device=self.device
        )
        new_models = self.switching_dyn(x_t_1[:, :, 1:], t)
        index = new_models[:, :, 0].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = (scaling * x_t_1[:, :, 0] + bias).unsqueeze(2) + noise
        return pt.cat((new_pos, new_models), dim=2)

    def log_eta_t(self, x_t, t: int):
        """
        辅助权重（Bootstrap滤波不使用）
        
        在Guided滤波中使用，用于调整提议分布
        """
        pass

    def log_R_0(self, x_0):
        """
        时间0的Radon-Nikodym导数
        
        用于重要性采样，Bootstrap滤波中为0
        """
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        """
        时间t的Radon-Nikodym导数
        
        用于重要性采样，计算模型切换的概率
        """
        return self.switching_dyn.get_log_probs(x_t[:, :, 1:], x_t_1[:, :, 1:])

    def log_f_t(self, x_t, t: int):
        """
        观测似然
        
        参数说明：
        ----------
        x_t: Tensor
            当前状态
        t: int
            当前时间步
        
        返回：
        ----------
        Tensor: 观测似然的对数
        
        说明：
        - 根据当前模型索引选择对应的a和b
        - 计算观测位置的预测：locs = a[k] * sqrt(|x_t|) + b[k]
        - 计算高斯似然：log p(y_t | x_t) ∝ -((y_t - locs)^2) / (2 * var_s)
        """
        index = x_t[:, :, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        locs = scaling * pt.sqrt(pt.abs(x_t[:, :, 0]) + 1e-7) + bias
        return self.var_factor * ((self.y[t] - locs) ** 2)

    def observation_generation(self, x_t):
        """
        从状态生成观测
        
        参数说明：
        ----------
        x_t: Tensor
            当前状态
        
        返回：
        ----------
        Tensor: 生成的观测值
        
        说明：
        - 用于模拟数据生成
        - y_t = a[k] * sqrt(|x_t|) + b[k] + noise
        """
        noise = self.y_dist.sample((x_t.size(0), 1)).to(device=self.device)
        index = x_t[:, :, 1].to(int)
        scaling = self.a[index]
        bias = self.b[index]
        new_pos = (scaling * pt.sqrt(pt.abs(x_t[:, :, 0])) + bias).unsqueeze(2) + noise
        return new_pos


class IMMPF(SSM):
    """
    交互多模型粒子滤波（Interacting Multiple Model Particle Filter）
    
    传统的IMM-PF算法，用于对比
    特点：
    - 使用预定义的切换模型（Markov/Polya/Erlang）
    - 不需要训练，直接测试
    - 作为对比基准，验证DIMMPF的改进效果
    
    参数说明：
    ----------
    a: List[int]
        状态转移系数
    b: List[int]
        状态转移偏置
    var_s: float
        噪声方差
    switching_dyn: Module
        切换动力学模型
    device: str
        运行设备
    """

    def set_observations(self, get_observation: Callable, t: int):
        """设置观测值"""
        self.y = self.reindexed_array(
            t - 1, [get_observation(t - 1), get_observation(t)]
        )

    def __init__(
        self,
        a: List[int],
        b: List[int],
        var_s: float,
        switching_dyn: pt.nn.Module,
        device: str = "cuda",
    ):
        super().__init__(device)
        self.n_models = len(a)
        self.a = pt.tensor(a, device=device)
        self.b = pt.tensor(b, device=device)
        self.switching_dyn = switching_dyn
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        var_s = pt.tensor(var_s)
        self.var_factor = -1 / (2 * var_s)
        self.y_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.alg = self.PF_Type.Bootstrap
        self.var_factor = -1 / (2 * var_s + 1e-6)
        self.pre_factor = -(1 / 2) * (
            pt.log(var_s + 1e-6) + pt.log(pt.tensor(2 * pt.pi))
        )

    def M_0_proposal(self, k, batches: int, n_samples: int):
        """
        时间0的提议分布（模型k）
        
        参数说明：
        ----------
        k: int
            模型索引
        batches: int
            批量大小
        n_samples: int
            每批样本数
        
        返回：
        ----------
        Tensor: 初始状态，包含位置和辅助变量
        """
        self.zeros = pt.zeros(
            (batches, n_samples, self.n_models), device=self.device, dtype=bool
        )
        init_locs = (
            self.init_x_dist.sample([batches, n_samples])
            .to(device=self.device)
            .unsqueeze(2)
        )
        init_r = self.switching_dyn.R_0(batches, n_samples, k)
        return pt.cat((init_locs, init_r), dim=2)

    def M_t_proposal(self, k, x_t_1, t: int):
        """
        时间t的提议分布（模型k）
        
        参数说明：
        ----------
        k: int
            模型索引
        x_t_1: Tensor
            前一时刻的状态
        t: int
            当前时间步
        
        返回：
        ----------
        Tensor: 新状态，包含位置和辅助变量
        
        说明：
        - 使用模型k的系数a[k]和b[k]
        - x_t = a[k] * x_{t-1} + b[k] + noise
        """
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(
            device=self.device
        )
        scaling = self.a[k]
        bias = self.b[k]
        new_pos = (scaling * x_t_1[:, :, 0] + bias).unsqueeze(2) + noise
        r = self.switching_dyn.R_t(x_t_1[:, :, 1:], k)
        return pt.cat((new_pos, r), dim=2).detach()

    def log_M_t(self, k, x_t, x_t_1, t: int):
        """
        提议分布的对数密度
        
        计算状态转移的对数概率密度
        """
        scaling = self.a[k]
        bias = self.b[k]
        locs = scaling * x_t_1[:, :, 0] + bias
        return self.var_factor * ((x_t[:, :, 0] - locs) ** 2) + self.pre_factor

    def log_eta_t(self, x_t, t: int):
        """辅助权重"""
        pass

    def log_R_0(self, x_0):
        """时间0的Radon-Nikodym导数"""
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        """
        时间t的Radon-Nikodym导数
        
        用于重要性采样校正
        """
        prop_density = self.log_M_t(x_t, x_t_1, t)
        return (
            self.switching_dyn.get_weight(x_t[:, :, 1:], x_t_1[:, :, 1:])
            + prop_density
            - prop_density.detach()
        )

    def log_f_t(self, k, x_t, t: int):
        """
        观测似然（模型k）
        
        计算在模型k下的观测似然
        """
        scaling = self.a[k]
        bias = self.b[k]
        locs = scaling * pt.sqrt(pt.abs(x_t[:, :, 0]) + 1e-7) + bias
        return self.var_factor * ((self.y[t] - locs) ** 2) + self.pre_factor

    def get_regime_probs(self, x_t):
        """获取模型概率"""
        return self.switching_dyn.get_regime_probs(x_t[:, :, 1:])


class RLPF(SSM):
    """
    递归线性粒子滤波（Recursive Linear Particle Filter）
    
    使用神经网络学习切换动力学和观测模型
    特点：
    - 使用神经网络学习切换动力学（替代预定义模型）
    - 保持状态转移方程的线性结构
    - 端到端训练所有参数
    
    与IMMPF的区别：IMMPF使用预定义切换模型，RLPF使用神经网络学习
    与DIMMPF的关系：RLPF是DIMMPF的简化版，只学习切换，状态转移保持线性
    
    参数说明：
    ----------
    n_models: int
        模型数量
    switching_dyn: Module
        切换动力学（神经网络）
    init_scale: float
        初始化缩放
    layers: int
        神经网络层数
    hidden_size: int
        隐藏层大小
    dyn: str
        动态类型
    device: str
        运行设备
    """

    def set_observations(self, get_observation: Callable, t: int):
        """设置观测值"""
        self.y = self.reindexed_array(
            t - 1, [get_observation(t - 1), get_observation(t)]
        )

    def __init__(
        self,
        n_models,
        switching_dyn: pt.nn.Module,
        init_scale=1,
        layers=2,
        hidden_size=8,
        dyn="Boot",
        device: str = "cuda",
    ):
        super().__init__(device)
        self.n_models = n_models
        
        # 为每个模型创建动态网络和观测网络
        # 动态网络：预测下一时刻状态
        self.dyn_models = pt.nn.ModuleList(
            [Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)]
        )
        # 观测网络：从状态预测观测
        self.obs_models = pt.nn.ModuleList(
            [Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)]
        )
        self.switching_dyn = switching_dyn
        
        # 参数初始化
        for p in self.parameters():
            p = p * init_scale
        
        # 可学习的噪声标准差
        # sd_d: 动态噪声（状态转移噪声）
        # sd_o: 观测噪声
        self.sd_d = pt.nn.Parameter(pt.rand(1) * 0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1) * 0.4 + 0.1)

        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.pi_fact = (1 / 2) * pt.log(pt.tensor(2 * pt.pi))
        
        if dyn == "Boot":
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def set_x_scaling(self, loc, scale):
        """
        设置状态缩放参数
        
        用于数据标准化
        """
        self.x_scale = scale
        self.x_loc = loc

    def M_0_proposal(self, batches: int, n_samples: int):
        """
        时间0的提议分布
        
        参数说明：
        ----------
        batches: int
            批量大小
        n_samples: int
            每批样本数
        
        返回：
        ----------
        Tensor: 初始状态，包含标准化后的位置和模型索引
        """
        self.zeros = pt.zeros(
            (batches, n_samples, self.n_models), device=self.device, dtype=bool
        )
        
        # 预计算因子
        self.var_factor = -1 / (2 * (self.sd_o**2) + 1e-6)
        self.pre_factor = -(1 / 2) * pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
        self.var_factor_dyn = -1 / (2 * (self.sd_d**2) + 1e-6)
        self.pre_factor_dyn = -(1 / 2) * pt.log(self.sd_d**2 + 1e-6) - self.pi_fact
        
        # 初始化位置和模型
        init_locs = (
            self.init_x_dist.sample([batches, n_samples])
            .to(device=self.device)
            .unsqueeze(2)
            - self.x_loc
        ) / self.x_scale
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        self.scatter = pt.scatter(self.zeros, 2, init_regimes.to(int), True)
        
        return pt.cat((init_locs, init_regimes), dim=2).detach()

    def M_t_proposal(self, x_t_1, t: int):
        """
        时间t的提议分布
        
        参数说明：
        ----------
        x_t_1: Tensor
            前一时刻的状态
        t: int
            当前时间步
        
        返回：
        ----------
        Tensor: 新状态，包含位置和模型索引
        
        说明：
        - 根据切换模型选择新模型
        - 根据模型索引选择对应的动态网络
        - 计算新位置：x_t = dyn_model(x_{t-1}) + noise
        """
        noise = (
            self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device)
            * self.sd_d
        )
        new_models = self.switching_dyn(x_t_1[:, :, 1:], t)
        locs = pt.empty((x_t_1.size(0), x_t_1.size(1)), device=self.device)
        index = new_models[:, :, 0:1].to(int)
        self.scatter = pt.scatter(self.zeros, 2, index, True)
        
        # 根据模型索引选择对应的动态网络
        for m in range(self.n_models):
            mask = self.scatter[:, :, m]
            locs[mask] = self.dyn_models[m](x_t_1[:, :, 0:1][mask]).squeeze()
        
        self.locs = locs
        new_pos = locs.unsqueeze(2) + noise
        return pt.cat((new_pos, new_models), dim=2).detach()

    def log_M_t(self, x_t, x_t_1, t: int):
        """提议分布的对数密度"""
        return (
            self.var_factor_dyn * ((x_t[:, :, 0] - self.locs) ** 2)
            + self.pre_factor_dyn
        )

    def log_eta_t(self, x_t, t: int):
        """辅助权重"""
        pass

    def log_R_0(self, x_0):
        """时间0的Radon-Nikodym导数"""
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        """
        时间t的Radon-Nikodym导数
        
        用于重要性采样校正
        """
        prop_density = self.log_M_t(x_t, x_t_1, t)
        return (
            self.switching_dyn.get_weight(x_t[:, :, 1:], x_t_1[:, :, 1:])
            + prop_density
            - prop_density.detach()
        )

    def log_f_t(self, x_t, t: int):
        """
        观测似然
        
        参数说明：
        ----------
        x_t: Tensor
            当前状态
        t: int
            当前时间步
        
        返回：
        ----------
        Tensor: 观测似然的对数
        
        说明：
        - 根据模型索引选择对应的观测网络
        - 计算观测预测：y_pred = obs_model(x_t)
        - 计算高斯似然
        """
        locs = pt.empty((x_t.size(0), x_t.size(1)), device=self.device)
        
        # 根据模型索引选择对应的观测网络
        for m in range(self.n_models):
            mask = self.scatter[:, :, m]
            locs[mask] = self.obs_models[m](x_t[:, :, 0:1][mask]).squeeze()
        
        return self.var_factor * ((self.y[t] - locs) ** 2) + self.pre_factor


class Redefined_RLPF(SSM):
    """
    重定义的RLPF（用于ELBO计算）
    
    与RLPF共享参数，但用于计算ELBO损失
    特点：
    - 共享RLPF的参数（动态网络、观测网络、切换模型）
    - 不同的前向传播逻辑，专门用于ELBO计算
    - 预计算似然，加速ELBO计算
    
    参数说明：
    ----------
    parent: RLPF
        父RLPF模型，共享其参数
    device: str
        运行设备
    """

    def set_observations(self, get_observation: Callable, t: int):
        """设置观测值"""
        self.y = self.reindexed_array(
            t - 1, [get_observation(t - 1), get_observation(t)]
        )

    def __init__(self, parent: RLPF, device: str = "cuda"):
        super().__init__(device)
        self.n_models = parent.n_models
        self.dyn_models = parent.dyn_models
        self.obs_models = parent.obs_models
        self.switching_dyn = parent.switching_dyn
        self.sd_d = parent.sd_d
        self.sd_o = parent.sd_o
        self.pi_fact = (1 / 2) * pt.log(pt.tensor(2 * pt.pi))
        self.alg = self.PF_Type.Bootstrap

    def set_up(self, state, observations):
        """
        预计算似然（用于ELBO）
        
        参数说明：
        ----------
        state: Tensor
            真实状态序列
        observations: Tensor
            观测序列
        
        说明：
        - 为每个模型预计算动态似然和观测似然
        - 动态似然：p(x_t | x_{t-1}, k)
        - 观测似然：p(y_t | x_t, k)
        """
        var_factor = -1 / (2 * (self.sd_o**2) + 1e-6)
        pre_factor = -(1 / 2) * pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
        var_factor_dyn = -1 / (2 * (self.sd_d**2) + 1e-6)
        pre_factor_dyn = -(1 / 2) * pt.log(self.sd_d**2 + 1e-6) - self.pi_fact
        
        self.dyn_probs_list = [None] * self.n_models
        self.likelihoods = [None] * self.n_models
        
        # 为每个模型预计算似然
        for k in range(self.n_models):
            locs_d = self.dyn_models[k](state[:, :-1, :])
            locs_o = self.obs_models[k](state)
            probs_d = (
                var_factor_dyn * ((state[:, 1:, :] - locs_d) ** 2) + pre_factor_dyn
            )
            likelihood = var_factor * (observations - locs_o) ** 2 + pre_factor
            likelihood[:, 1:, :] = likelihood[:, 1:, :] + probs_d
            self.likelihoods[k] = likelihood.squeeze()

    def M_0_proposal(self, k, batches: int, n_samples: int):
        """
        时间0的提议分布
        
        参数说明：
        ----------
        k: int
            模型索引
        batches: int
            批量大小
        n_samples: int
            每批样本数
        """
        self.zeros = pt.zeros(
            (batches, n_samples, self.n_models), device=self.device, dtype=bool
        )
        init_r = self.switching_dyn.R_0(batches, n_samples, k)
        return init_r

    def M_t_proposal(self, k, x_t_1, t: int):
        """
        时间t的提议分布
        
        使用切换模型更新辅助变量
        """
        r = self.switching_dyn.R_t(x_t_1, k)
        return r

    def log_M_t(self, k, x_t, x_t_1, t: int):
        """
        提议分布的对数密度
        
        对于ELBO计算，提议分布是均匀的，返回0
        """
        return pt.zeros((x_t.size(0), x_t.size(1), 1), device=self.device)

    def log_eta_t(self, x_t, t: int):
        """辅助权重"""
        pass

    def log_f_t(self, k, x_t, t: int):
        """
        观测似然（使用预计算的似然）
        
        从预计算的likelihoods中查找模型k在时间t的似然
        """
        return self.likelihoods[k][:, t : t + 1].expand(-1, x_t.size(1))

    def get_regime_probs(self, x_t):
        """获取模型概率"""
        return self.switching_dyn.get_regime_probs(x_t)


# =============================================================================
# 第四部分：基线模型
# LSTM和Transformer用于对比
# =============================================================================

class LSTM(pt.nn.Module):
    """
    LSTM基线模型
    
    使用LSTM网络直接映射观测到状态
    作为对比基准，验证粒子滤波方法的优势
    
    参数说明：
    ----------
    obs_dim: int
        观测维度
    hid_dim: int
        隐藏层维度
    state_dim: int
        状态维度
    n_layers: int
        LSTM层数
    device: str
        运行设备
    """

    def __init__(self, obs_dim, hid_dim, state_dim, n_layers, device="cuda") -> None:
        super().__init__()
        # LSTM参数说明：
        # obs_dim: 输入维度
        # hid_dim: 隐藏层维度
        # n_layers: LSTM层数
        # True: batch_first（输入形状为(batch, seq, feature)）
        # True: bidirectional（双向LSTM）
        # 0.0: dropout（不使用dropout）
        # False: 不使用偏置
        # state_dim: 投影维度
        self.lstm = pt.nn.LSTM(
            obs_dim, hid_dim, n_layers, True, True, 0.0, False, state_dim, device
        )

    def forward(self, y_t):
        """
        前向传播
        
        参数说明：
        ----------
        y_t: Tensor
            观测序列，形状为(batch, seq, obs_dim)
        
        返回：
        ----------
        Tensor: 预测的状态序列，形状为(batch, seq, state_dim)
        """
        return self.lstm(y_t)[0]


class Transformer(pt.nn.Module):
    """
    Transformer基线模型
    
    使用Transformer网络直接映射观测到状态
    作为对比基准，验证粒子滤波方法的优势
    
    参数说明：
    ----------
    obs_dim: int
        观测维度
    hid_dim: int
        隐藏层维度
    state_dim: int
        状态维度
    T: int
        序列长度
    device: str
        运行设备
    layers: int
        Transformer层数
    """

    def __init__(
        self, obs_dim, hid_dim, state_dim, T: int = 50, device="cuda", layers=2
    ):
        super().__init__()
        # Transformer编码器层
        self.encoder_layer = pt.nn.TransformerEncoderLayer(
            hid_dim,  # 隐藏层维度
            1,  # 注意力头数
            hid_dim,  # 前馈网络维度
            0.1,  # dropout率
            batch_first=True,  # 输入形状为(batch, seq, feature)
            device=device
        )
        # Transformer编码器
        self.transformer = pt.nn.TransformerEncoder(self.encoder_layer, layers)
        # 编码层：将观测映射到隐藏空间
        self.encoding = pt.nn.Linear(obs_dim, hid_dim, device=device)
        # 解码层：将隐藏状态映射到状态空间
        self.decoding = pt.nn.Linear(hid_dim, state_dim, device=device)
        self.relu = pt.nn.ReLU()
        # 因果掩码（防止看到未来信息）
        # 使用下三角矩阵，确保位置i只能看到位置0到i的信息
        self.mask = pt.tril(pt.ones((T + 1, T + 1), device=device))

    def forward(self, y_t):
        """
        前向传播
        
        参数说明：
        ----------
        y_t: Tensor
            观测序列，形状为(batch, seq, obs_dim)
        
        返回：
        ----------
        Tensor: 预测的状态序列，形状为(batch, seq, state_dim)
        
        说明：
        1. 编码：将观测映射到隐藏空间
        2. ReLU激活
        3. Transformer编码（使用因果掩码防止看到未来）
        4. 解码：将隐藏状态映射到状态空间
        """
        t = self.encoding(y_t)
        t = self.relu(t)
        t = self.transformer(t, mask=self.mask, is_causal=True)
        return self.decoding(t)

    def set_up(self, state, observations):
        """
        预计算似然（用于ELBO）
        
        参数说明：
        ----------
        state: Tensor
            真实状态序列
        observations: Tensor
            观测序列
        
        说明：
        - 为每个模型预计算动态似然和观测似然
        - 动态似然：p(x_t | x_{t-1}, k)
        - 观测似然：p(y_t | x_t, k)
        - 总似然：p(x_t, y_t | x_{t-1}, k) = p(x_t | x_{t-1}, k) * p(y_t | x_t, k)
        """
        var_factor = -1 / (2 * (self.sd_o**2) + 1e-6)
        pre_factor = -(1 / 2) * pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
        var_factor_dyn = -1 / (2 * (self.sd_d**2) + 1e-6)
        pre_factor_dyn = -(1 / 2) * pt.log(self.sd_d**2 + 1e-6) - self.pi_fact
        
        self.dyn_probs_list = [None] * self.n_models
        self.likelihoods = [None] * self.n_models
        
        # 为每个模型预计算似然
        for k in range(self.n_models):
            locs_d = self.dyn_models[k](state[:, :-1, :])
            locs_o = self.obs_models[k](state)
            probs_d = (
                var_factor_dyn * ((state[:, 1:, :] - locs_d) ** 2) + pre_factor_dyn
            )
            likelihood = var_factor * (observations - locs_o) ** 2 + pre_factor
            likelihood[:, 1:, :] = likelihood[:, 1:, :] + probs_d
            self.likelihoods[k] = likelihood.squeeze()

    def M_0_proposal(self, batches: int, n_samples: int):
        """时间0的提议分布"""
        init_regimes = self.switching_dyn.init_state(batches, n_samples)
        self.zeros = pt.zeros(
            (batches, n_samples, self.n_models), device=self.device, dtype=bool
        )
        self.scatter = pt.scatter(self.zeros, 2, init_regimes.to(int), True)
        return init_regimes

    def M_t_proposal(self, x_t_1, t: int):
        """时间t的提议分布"""
        new_models = self.switching_dyn(x_t_1, t)
        return new_models

    def log_eta_t(self, x_t, t: int):
        """辅助权重"""
        pass

    def log_R_0(self, x_0):
        """时间0的Radon-Nikodym导数"""
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        """时间t的Radon-Nikodym导数"""
        return self.switching_dyn.get_weight(x_t[:, :, 1:], x_t_1[:, :, 1:])

    def log_f_t(self, x_t, t: int):
        """
        观测似然（使用预计算的似然）
        
        从预计算的likelihoods中查找对应模型的似然
        """
        index = x_t[:, :, 0:1].to(int)
        self.scatter = pt.scatter(self.zeros, 2, index, True)
        probs = pt.empty((x_t.size(0), x_t.size(1)), device=self.device)
        
        for m in range(self.n_models):
            mask = self.scatter[:, :, m]
            probs[mask] = (self.likelihoods[m][:, t, None].expand(-1, mask.size(1)))[
                mask
            ]
        return probs


class DIMMPF(SSM):
    """
    深度交互多模型粒子滤波（Deep Interacting Multiple Model Particle Filter）
    
    论文的核心算法，使用神经网络学习切换动力学和状态转移
    特点：
    - 端到端学习切换动力学（使用NN_Switching）
    - 端到端学习状态转移（使用神经网络替代线性方程）
    - 可微分粒子滤波，支持梯度反向传播
    
    与RLPF的区别：
    - RLPF：学习切换，状态转移保持线性
    - DIMMPF：同时学习切换和状态转移（都是神经网络）
    
    参数说明：
    ----------
    n_models: int
        模型数量
    switching_dyn: Module
        切换动力学（神经网络）
    init_scale: float
        初始化缩放
    layers: int
        神经网络层数
    hidden_size: int
        隐藏层大小
    dyn: str
        动态类型
    device: str
        运行设备
    """

    def set_observations(self, get_observation: Callable, t: int):
        """设置观测值"""
        self.y = self.reindexed_array(
            t - 1, [get_observation(t - 1), get_observation(t)]
        )

    def __init__(
        self,
        n_models,
        switching_dyn: pt.nn.Module,
        init_scale=1,
        layers=2,
        hidden_size=8,
        dyn="Boot",
        device: str = "cuda",
    ):
        super().__init__(device)
        self.n_models = n_models
        
        # 为每个模型创建动态网络和观测网络
        # 与RLPF类似，但DIMMPF使用IMM框架（多模型并行）
        self.dyn_models = pt.nn.ModuleList(
            [Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)]
        )
        self.obs_models = pt.nn.ModuleList(
            [Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)]
        )
        self.switching_dyn = switching_dyn
        
        # 参数初始化
        for p in self.parameters():
            p = p * init_scale
        
        # 可学习的噪声标准差
        self.sd_d = pt.nn.Parameter(pt.rand(1) * 0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1) * 0.4 + 0.1)

        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        self.pi_fact = (1 / 2) * pt.log(pt.tensor(2 * pt.pi))
        self.alg = self.PF_Type.Bootstrap

    def set_x_scaling(self, loc, scale):
        """设置状态缩放参数"""
        self.x_scale = scale
        self.x_loc = loc

    def M_0_proposal(self, k, batches: int, n_samples: int):
        """
        时间0的提议分布（模型k）
        
        参数说明：
        ----------
        k: int
            模型索引
        batches: int
            批量大小
        n_samples: int
            每批样本数
        """
        self.zeros = pt.zeros(
            (batches, n_samples, self.n_models), device=self.device, dtype=bool
        )
        
        # 预计算因子
        self.var_factor = -1 / (2 * (self.sd_o**2) + 1e-6)
        self.pre_factor = -(1 / 2) * pt.log(self.sd_o**2 + 1e-6) - self.pi_fact
        self.var_factor_dyn = -1 / (2 * (self.sd_d**2) + 1e-6)
        self.pre_factor_dyn = -(1 / 2) * pt.log(self.sd_d**2 + 1e-6) - self.pi_fact
        
        # 初始化
        init_locs = (
            self.init_x_dist.sample([batches, n_samples])
            .to(device=self.device)
            .unsqueeze(2)
            - self.x_loc
        ) / self.x_scale
        init_r = self.switching_dyn.R_0(batches, n_samples, k)
        return pt.cat((init_locs, init_r), dim=2)

    def M_t_proposal(self, k, x_t_1, t: int):
        """
        时间t的提议分布（模型k）
        
        使用模型k的动态网络预测下一时刻状态
        """
        noise = (
            self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(device=self.device)
            * self.sd_d
        )
        locs = self.dyn_models[k](x_t_1[:, :, 0:1])
        new_pos = locs + noise
        r = self.switching_dyn.R_t(x_t_1[:, :, 1:], k)
        return pt.cat((new_pos, r), dim=2)

    def log_M_t(self, k, x_t, x_t_1, t: int):
        """
        提议分布的对数密度（模型k）
        
        计算状态转移的对数概率密度
        """
        locs = self.dyn_models[k](x_t_1[:, :, 0:1]).squeeze()
        locs = locs[:, None, :]
        return (
            self.var_factor_dyn * ((x_t[:, :, None, 0] - locs) ** 2)
            + self.pre_factor_dyn
        )

    def log_eta_t(self, x_t, t: int):
        """辅助权重"""
        pass

    def log_f_t(self, k, x_t, t: int):
        """
        观测似然（模型k）
        
        使用模型k的观测网络计算观测似然
        """
        locs = self.obs_models[k](x_t[:, :, 0:1])
        return (
            self.var_factor * ((self.y[t][:, None, :] - locs) ** 2) + self.pre_factor
        ).squeeze()

    def get_regime_probs(self, x_t):
        """获取模型概率"""
        return self.switching_dyn.get_regime_probs(x_t[:, :, 1:])


class DIMMPF_redefined(SSM):
    """
    重定义的DIMMPF（用于ELBO计算）
    
    与DIMMPF共享参数，但用于计算ELBO损失
    特点：
    - 共享DIMMPF的参数（动态网络、观测网络、切换模型）
    - 不同的前向传播逻辑，专门用于ELBO计算
    - 预计算似然，加速ELBO计算
    
    参数说明：
    ----------
    parent: DIMMPF
        父DIMMPF模型，共享其参数
    device: str
        运行设备
    """

    def set_observations(self, get_observation: Callable, t: int):
        """设置观测值"""
        self.y = self.reindexed_array(
            t - 1, [get_observation(t - 1), get_observation(t)]
        )

    def __init__(self, parent: DIMMPF, device: str = "cuda"):
        super().__init__(device)
        self.n_models = parent.n_models
        self.dyn_models = parent.dyn_models
        self.obs_models = parent.obs_models
        self.switching_dyn = parent.switching_dyn
        self.sd_d = parent.sd_d
        self.sd_o = parent.sd_o
        self.pi_fact = (1 / 2) * pt.log(pt.tensor(2 * pt.pi))
        self.alg = self.PF_Type.Bootstrap
