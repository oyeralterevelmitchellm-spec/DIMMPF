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
            # x_t_1 形状说明: (batches, n_samples, state_dim)
            #   - x_t_1.size(0) = batches: 批次大小，同时处理的独立序列数（如2个目标轨迹）
            #   - x_t_1.size(1) = n_samples: 每批次的粒子数量（如96个粒子）
            #   - x_t_1.size(2) = state_dim: 状态维度（这里是模型索引，值为1）
            #
            # .tile((x_t_1.size(0), x_t_1.size(1) // self.n_models)) 说明:
            #   - 示例: batches=2, n_samples=96, n_models=8
            #   - 第0维重复2次（对应2个批次）
            #   - 第1维重复 96//8=12 次（每个模型分配12个粒子）
            #   - 目的: 均匀分配粒子到各模型，每个模型获得相同数量粒子
            #   - 要求: n_samples 必须能被 n_models 整除
            return (
                pt.arange(self.n_models, device=self.device)
                .tile((x_t_1.size(0), x_t_1.size(1) // self.n_models))
                .unsqueeze(2)
            )
        
        # 随机切换：根据转移概率采样
        # shifts 生成详解：
        #   1. pt.multinomial(self.probs, x_t_1.size(0) * x_t_1.size(1), True):
        #      - self.probs: 转移概率分布，形状 (n_models,)，表示从当前模型转移到其他模型的概率
        #      - x_t_1.size(0) * x_t_1.size(1): 采样次数 = 批次大小 × 每批粒子数
        #      - True: 允许重复采样
        #      - 作用: 根据马尔可夫转移概率，为每个粒子随机采样一个"偏移量"
        #      - 示例: probs=[0.7,0.2,0.05,0.05]，采样返回 [0,0,1,0,2,0,...]，70%是0，20%是1
        #
        #   2. .to(self.device): 将张量移动到指定设备（GPU/CPU），确保设备一致性
        #
        #   3. .reshape([x_t_1.size(0), x_t_1.size(1)]):
        #      - 将一维采样结果 (192,) 重塑为二维 (batches, n_samples)，如 (2, 96)
        #      - 匹配粒子状态形状，便于后续逐元素运算
        #
        #   物理意义: shifts 表示每个粒子要"跳转"多少个模型位置
        #      - shifts=0: 保持当前模型（概率最高）
        #      - shifts=1: 切换到下一个模型
        #      - shifts=k: 向前跳转k个模型（循环）
        #
        # 通俗理解：
        #   - 想象有192个粒子（2批×96个），每个粒子决定：换不换模型？换到哪个？
        #   - pt.multinomial 按概率随机选择（如70%概率不换，20%概率换到下一个）
        #   - .reshape 把192个数字排成2行96列，对应2批粒子
        shifts = (
            pt.multinomial(self.probs, x_t_1.size(0) * x_t_1.size(1), True)
            .to(self.device)
            .reshape([x_t_1.size(0), x_t_1.size(1)])
        )
        # new_models 计算详解：
        #   - x_t_1[:, :, 0]: 粒子当前在哪个模型（如模型3）
        #     x_t_1 是前一时刻的粒子状态，形状 (batches, n_samples, state_dim)
        #     在本项目中，state_dim 的第0维就是模型编号（model index）
        #     例如：x_t_1 = [[[3, 0.5], [5, 0.2]]] 表示2个粒子，粒子0在模型3，粒子1在模型5
        #   - shifts: 要跳几步（如跳2步）
        #   - +: 3 + 2 = 5，新模型是模型5
        #   - pt.remainder(..., n_models): 如果超过最大模型号，就循环回来（取余）
        #     例如：7 + 2 = 9，9 % 8 = 1，回到模型1
        #   - 效果：实现模型间的循环跳转（0→1→2→...→7→0→1...）
        new_models = pt.remainder(shifts + x_t_1[:, :, 0], self.n_models)
        return new_models.unsqueeze(2)

    def get_log_probs(self, x_t, x_t_1):
        """
        计算切换的对数概率

        用于计算粒子权重，根据模型切换计算对数概率
        """
        # 计算实际切换的偏移量：当前模型 - 上一时刻模型
        # 示例: x_t=[3,5,1], x_t_1=[1,5,0] → shifts=[2,0,1]
        #   - 2: 从模型1切换到模型3（向前跳2步）
        #   - 0: 保持在模型5（没切换）
        #   - 1: 从模型0切换到模型1（向前跳1步）
        shifts = x_t[:, :, 0] - x_t_1[:, :, 0]
        # 取模确保偏移量在有效范围内（0~n_models-1），处理循环情况
        # 示例: -1%8=7, 8%8=0，实现模型循环（7→0→1...）
        shifts = pt.remainder(shifts, self.n_models).to(int)
        
        # 详细解释：
        #   - switching_vec 是在 __init__ 中定义的 pt.log(tprobs)
        #   - 假设 n_models=8，转移概率 tprobs = [0.7, 0.2, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005]
        #   - 则 switching_vec = [-0.36, -1.61, -3.00, -3.91, -4.61, -4.61, -5.30, -5.30]
        #
        #   偏移量与概率对应关系：
        #     0: 保持当前模型（概率0.7，最可能）
        #     1: 跳1步（概率0.2）
        #     2: 跳2步（概率0.05）
        #     ...
        #     7: 跳7步（概率0.005，最不可能）
        #
        #   执行过程（高级索引）：
        #     shifts = [[0, 2, 1],      # 第1批3个粒子
        #               [1, 0, 7]]      # 第2批3个粒子
        #     查表结果:
        #       [0] → -0.36  (保持，概率最高)
        #       [2] → -3.00  (跳2步)
        #       [1] → -1.61  (跳1步)
        #       [1] → -1.61
        #       [0] → -0.36
        #       [7] → -5.30  (跳7步，概率最低)
        #     返回: [[-0.36, -3.00, -1.61],
        #            [-1.61, -0.36, -5.30]]
        #
        #   物理意义：
        #     - 偏移量越小（保持模型），对数概率越大，粒子权重增加越多
        #     - 偏移量越大（跳很多步），对数概率越小，粒子权重增加越少
        #     - 这样保持模型的粒子获得更高权重，符合马尔可夫假设
        return self.switching_vec[shifts]

    def get_regime_probs(self, x_t_1):
        """
        获取模型概率分布

        用于IMM算法中的模型概率混合，计算每个模型的对数概率
        """
        # 创建所有可能的模型索引 [0, 1, 2, ..., n_models-1]
        ks = pt.arange(0, self.n_models, device=self.device)
        # 计算从当前模型到所有可能模型的偏移量
        # ks[None, None, :]: 将形状从 (8,) 扩展为 (1, 1, 8)，用于广播  （None 的作用 ：在张量中插入一个大小为1的维度。）
        # x_t_1[:, :, 0:1]: 取当前模型索引，形状 (100, 200, 1)
        # 广播相减后形状: (100, 200, 8)，表示每个粒子到8个目标模型的偏移
        #
        # 示例 (n_models=8, 2个粒子):
        #   ks = [0, 1, 2, 3, 4, 5, 6, 7]
        #   粒子0在模型3: [0-3, 1-3, ..., 7-3] = [-3, -2, -1, 0, 1, 2, 3, 4]
        #   粒子1在模型5: [0-5, 1-5, ..., 7-5] = [-5, -4, -3, -2, -1, 0, 1, 2]
        #   含义: 到模型0需-3步(或+5步)，到模型3需0步，到模型7需+4步
        shifts = ks[None, None, :] - x_t_1[:, :, 0:1]
        # 取模将负偏移转为正数，实现循环 (如 -3%8=5)
        shifts = pt.remainder(shifts, self.n_models).to(int)
        # 查表获取对数概率，reshape保持形状 (100, 200, 8)
        # 返回每个粒子到8个模型的转移对数概率，用于IMM模型混合
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

        详细说明：
        ----------
        创建形状为 (batches, n_samples, 1) 的张量，所有元素都是 k
        用于 IMM 算法中标识粒子属于哪个模型

        示例 (batches=2, n_samples=3, k=3):
            返回: [[[3], [3], [3]],    # 第1批，3个粒子，辅助变量都是3
                   [[3], [3], [3]]]    # 第2批，3个粒子，辅助变量都是3
            形状: (2, 3, 1)
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

        详细说明：
        ----------
        创建与 r_t_1 形状相同的全1张量，然后乘以 k
        与 R_0 的区别：R_0 自己指定形状，R_t 根据已有张量形状创建

        示例 (r_t_1 形状 (2, 3, 1), k=5):
            返回: [[[5], [5], [5]],
                   [[5], [5], [5]]]
            所有粒子的辅助变量更新为 5
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

        详细说明：
        ----------
        初始化粒子状态，适用于 Polya（波利亚）切换或需要计数器的场景。

        执行步骤：
        1. 创建 scatter_v 全0张量，形状 (batches, n_samples, n_models)
           用于后续记录每个粒子选择各模型的次数

        2. 随机初始化模型索引 i_models：
           - pt.multinomial(self.ones_vec, batches*n_samples, True)
             作用：从 n_models 个模型中随机抽取 batches*n_samples 次
             示例 (batches=2, n_samples=96, n_models=8)：
               从 [0,1,2,3,4,5,6,7] 中随机抽192次，返回 [3,7,1,0,4,2,5,6,3,1,...]
               第1个粒子→模型3，第2个粒子→模型7，第3个粒子→模型1...
           - .reshape((batches, n_samples, 1))：重塑为 (2, 96, 1)

        3. 拼接模型索引和计数器：
           - i_models: (batches, n_samples, 1)，模型索引
           - ones: (batches, n_samples, n_models)，8个模型初始计数都是1
           - pt.concat(..., dim=2)：在第2维拼接，结果形状 (batches, n_samples, 9)

        返回结果示例 (n_models=8)：
            [[[3, 1, 1, 1, 1, 1, 1, 1, 1],   # 粒子0：模型3，8个模型计数都是1
              [7, 1, 1, 1, 1, 1, 1, 1, 1],   # 粒子1：模型7
              [1, 1, 1, 1, 1, 1, 1, 1, 1],   # 粒子2：模型1
              ...],
             ...]
            形状: (batches, n_samples, 9)
            第0维：当前模型索引
            第1-8维：模型0-7的选择次数计数器（用于Polya切换）
        """
        # 创建全0张量，用于记录每个粒子选择各模型的次数
        self.scatter_v = pt.zeros(
            (batches, n_samples, self.n_models), device=self.device
        )
        # 随机初始化模型索引
        # pt.multinomial: 从 n_models 个模型中均匀随机抽取 batches*n_samples 次
        # 示例: 返回 [3,7,1,0,4,2,5,6,3,1,...]，表示每个粒子初始在哪个模型
        i_models = (
            pt.multinomial(self.ones_vec, batches * n_samples, True)
            .reshape((batches, n_samples, 1))
            .to(device=self.device)
        )
        # 拼接模型索引和计数器（初始计数都是1）
        # 返回形状: (batches, n_samples, 1+n_models)
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
        # 将 scatter_v 清零，准备记录当前选择的模型
        self.scatter_v.zero_()
        # 在 scatter_v 中标记当前选中的模型位置为1
        # 详细解释：
        #   - x_t_1[:, :, 0]: 取出当前模型编号，形状 (batches, n_samples)
        #   - .unsqueeze(2): 增加维度，变成 (batches, n_samples, 1)
        #   - .to(int): 转成整数类型，用于索引
        #   - scatter_(2, ..., 1): 在第2维（模型维度），根据索引填入1
        #
        # 示例（1个粒子，n_models=8，当前在模型3）：
        #   scatter_v 初始: [0, 0, 0, 0, 0, 0, 0, 0]
        #   索引: 3
        #   scatter_v 结果: [0, 0, 0, 1, 0, 0, 0, 0]  # 位置3变成1
        #
        # 比喻：在8个抽屉中，给这次选中的那个抽屉贴个标签"1"
        self.scatter_v.scatter_(2, x_t_1[:, :, 0].unsqueeze(2).to(int), 1)
        # 更新计数器：历史计数 + 当前标记 = 新计数
        # 详细解释：
        #   - x_t_1[:, :, 1:]: 历史计数器，形状 (batches, n_samples, n_models)
        #   - scatter_v: 当前标记，形状 (batches, n_samples, n_models)
        #   - +: 逐元素相加
        #
        # 示例（接上，历史计数都是1）：
        #   历史计数: [1, 1, 1, 1, 1, 1, 1, 1]  # 8个模型各被选1次
        #   当前标记: [0, 0, 0, 1, 0, 0, 0, 0]  # 这次选了模型3
        #   新计数 c: [1, 1, 1, 2, 1, 1, 1, 1]  # 模型3变成2次了！
        #
        # 比喻：给选中的抽屉加一个金币，让它下次更容易被选中
        # 这就是 Polya 切换的"富者愈富"效应
        c = x_t_1[:, :, 1:] + self.scatter_v
        
        if self.dyn == "Uni":
            # 均匀采样：完全随机选择新模型，不考虑历史计数
            # 详细解释：
            #   - pt.multinomial(self.ones_vec, ..., True):
            #     从 n_models 个模型中均匀随机抽取 batches*n_samples 次
            #     self.ones_vec = [1,1,1,1,1,1,1,1]，表示8个模型概率相同（各1/8）
            #     示例: 返回 [3,7,1,0,4,2,5,6,3,1,...]，每个粒子随机分配一个模型
            #   - .reshape([x_t_1.size(0), x_t_1.size(1), 1]): 重塑为 (100, 200, 1)
            #   - pt.concat((..., c), dim=2): 拼接新模型索引和计数器
            #     返回形状: (100, 200, 9)，第0维是新模型，后面8维是计数器
            #
            # "Uni" vs Polya 的区别：
            #   - "Uni": 完全随机，不看计数器（本分支）
            #   - Polya: 根据计数器 c 采样，选得越多越容易被选（else分支）
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
        # 详细解释：
        #   - c.reshape(-1, self.n_models): 将计数器展平为 (batches*n_samples, n_models)
        #     示例: c 形状 (100, 200, 8) → reshape → (20000, 8)
        #   - pt.multinomial(c, 1, True): 根据计数器采样，计数越高概率越大  （1：每行采样1次）
        #     示例: c = [1, 1, 1, 2, 1, 1, 1, 1]（模型3计数=2，其他=1）
        #           概率: [1/9, 1/9, 1/9, 2/9, 1/9, 1/9, 1/9, 1/9]
        #           模型3被选中的概率是其他模型的2倍
        #     返回: 3（最可能返回模型3）
        #
        # "富者愈富"效果演示：
        #   第1次: c=[1,1,1,1,1,1,1,1] → 随机选了模型3
        #   第2次: c=[1,1,1,2,1,1,1,1] → 模型3概率更高，更可能被选
        #   第3次: c=[1,1,1,3,1,1,1,1] → 概率更高...
        #   模型3被选得越多，越容易被选 → Polya切换的核心
        #
        # 与 "Uni" 模式的区别：
        #   - "Uni": pt.multinomial(self.ones_vec, ...) 均匀随机
        #   - Polya: pt.multinomial(c, ...) 根据计数器，富者愈富
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
        # 取出计数器（第1维到第n_models维）
        # x_t 形状: (batches, n_samples, 1+n_models)
        # x_t[:, :, 1:] 形状: (batches, n_samples, n_models)
        # 示例: [1, 1, 1, 2, 1, 1, 1, 1] 表示8个模型各被选了多少次
        probs = x_t[:, :, 1:]
        # 归一化：将计数变成概率（所有概率之和=1）
        # pt.sum(probs, dim=2, keepdim=True): 在第2维求和，keepdim=True：保持维度便于广播
        # 示例: [1, 1, 1, 2, 1, 1, 1, 1] 总和=9
        #       → [0.11, 0.11, 0.11, 0.22, 0.11, 0.11, 0.11, 0.11]
        # 模型3计数=2，概率=0.22，是其他模型的2倍
        probs /= pt.sum(probs, dim=2, keepdim=True)
        # 选择实际发生的那个模型的概率
        # x_t_1[:, :, 1]: 取出上一时刻实际选择的模型编号
        # .to(int): 转成整数索引
        # batched_select: 根据索引从probs中选出对应的概率
        # 示例: probs=[0.11,0.11,0.11,0.22,0.11,0.11,0.11,0.11]
        #       实际选了模型3 → s_probs = 0.22 

        #probs: [0.11, 0.11, 0.11, 0.22, 0.11, 0.11, 0.11, 0.11]
        #        ↑0    ↑1    ↑2    ↑3    ↑4    ↑5    ↑6    ↑7
        #假设 x_t_1[:, :, 1] = 3  （上一时刻实际选了模型3）
        #batched_select(probs, 3) = 0.22
        #s_probs = 0.22
        s_probs = batched_select(probs, x_t_1[:, :, 1].to(int))
        # 取对数，用于粒子权重计算
        # pt.log(0.22) = -1.51
        # 选得多的模型概率高，对数概率大（负得少），粒子权重就高
        return pt.log(s_probs)

    def get_regime_probs(self, x_t_1):
        """
        获取模型概率

        计算每个模型的概率（归一化计数器）
        """
        # 将计数器归一化，变成概率分布
        # x_t_1 形状: (batches, n_samples, 1+n_models)
        # 包含模型索引和各模型计数
        # pt.sum(x_t_1, dim=2, keepdim=True): 在第2维求和，保持维度便于广播
        #
        # 示例（1个粒子，n_models=8）:
        #   x_t_1 = [3, 1, 1, 1, 2, 1, 1, 1, 1]
        #           ↑ 第0维是模型3
        #              ↑ 后面8维是各模型计数
        #   计数: [1, 1, 1, 2, 1, 1, 1, 1]，总和=9
        #   归一化: [0.11, 0.11, 0.11, 0.22, 0.11, 0.11, 0.11, 0.11]
        #   模型3计数=2，概率=0.22，是其他模型的2倍
        probs = x_t_1 / pt.sum(x_t_1, dim=2, keepdim=True)
        # 取对数，返回所有模型的对数概率
        # 输出形状: (batches, n_samples, n_models)
        # 示例: [-2.20, -2.20, -2.20, -1.51, -2.20, -2.20, -2.20, -2.20]
        #        ↑0     ↑1     ↑2     ↑3     ↑4     ↑5     ↑6     ↑7
        #                              ↑
        #                            模型3对数概率最高（被选得多）
        #
        # 与 get_log_probs 的区别:
        #   - get_log_probs: 返回实际发生模型的对数概率（用于粒子权重）
        #   - get_regime_probs: 返回所有模型的对数概率（用于IMM模型混合）
        return pt.log(probs)

    def R_0(self, batches, n_samples, k):
        """
        初始化IMM辅助变量

        用于IMM（交互多模型）算法的模型概率计算。
        初始化时给模型k更高的计数（2），表示对模型k的初始偏向。

        参数说明：
        ----------
        batches: int
            批量大小
        n_samples: int
            每批样本数
        k: int
            模型索引（当前活跃的模型）

        返回：
        ----------
        Tensor: 初始化的辅助变量，形状 (batches, n_samples, n_models)
                模型k的计数为2，其他为1

        详细说明：
        ----------
        在IMM算法中，这个辅助变量用于计算模型混合概率。
        给模型k计数设为2，其他为1，表示"假设当前在模型k"的概率分布。

        示例（batches=1, n_samples=1, n_models=8, k=3）：
            t = [[[1, 1, 1, 1, 1, 1, 1, 1]]]  # 初始全1
            t[:, :, 3] = 2  # 模型3计数变2
            t = [[[1, 1, 1, 2, 1, 1, 1, 1]]]  # 模型3概率更高

        与Polya切换的关系：
            - Polya切换：用计数器选择新模型（富者愈富）
            - R_0/R_t：为IMM算法提供模型混合概率计算
        """
        # 创建全1张量，形状 (batches, n_samples, n_models)
        t = pt.ones((batches, n_samples, self.n_models), device=self.device)
        # 将模型k的计数设为2，表示对该模型的初始偏向
        # 在IMM中，这表示"当前假设在模型k"的概率分布
        t[:, :, k] = 2
        return t

    def R_t(self, r_t_1, k):
        """
        更新IMM辅助变量

        增加模型k的计数，用于IMM算法的模型概率更新。
        随着时间推移，持续增加模型k的计数，表示"在当前模型停留更长时间"的概率增加。

        参数说明：
        ----------
        r_t_1: Tensor
            上一时刻的辅助变量，形状 (batches, n_samples, n_models)
        k: int
            模型索引（要增加计数的模型）

        返回：
        ----------
        Tensor: 更新后的辅助变量，模型k的计数加1

        详细说明：
        ----------
        在IMM算法中，这个更新操作模拟Erlang分布的特性：
        模型在当前状态停留一段时间后才会切换。
        增加计数表示延长在当前模型的停留时间。

        示例（接上，k=3）：
            r_t_1 = [[[1, 1, 1, 2, 1, 1, 1, 1]]]  # 模型3计数=2
            temp[:, :, 3] += 1  # 模型3计数加1
            返回 = [[[1, 1, 1, 3, 1, 1, 1, 1]]]  # 模型3计数=3

        完整流程（Polya + IMM）：
            1. R_0(k): 初始化，假设当前在模型k
            2. Polya切换: 根据计数器选择新模型
            3. R_t(k): 更新辅助变量，计算IMM混合概率
            4. 结合两者进行状态估计
        """
        # 复制引用（注意：这里应该使用clone避免原地修改）
        temp = r_t_1
        # 增加模型k的计数，表示延长在当前模型的停留时间
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
        
        0.6/0.4 非对称切换概率的数学含义：
        ---------------------------------
        对应论文中的公式 20f，表示模式切换的非对称性：
        
            P(k_t = k_{t-1} + 1) = 0.6   # 前向切换（概率更高）
            P(k_t = k_{t-1} - 1) = 0.4   # 后向切换（概率更低）
        
        这种非对称设计的物理直觉：
            - 系统倾向于向前演化（如车辆加速、模式递进）
            - 后向切换是"回退"或"恢复"，概率较低
            - 0.6/0.4 的分配体现了这种方向偏置
        
        与 PF 状态转移的联系：
            - 在经典 PF 中：p(x_t | x_{t-1}) 是连续状态转移
            - 在 IMMPF 中：p(k_t | k_{t-1}) 是离散模式转移
            - 这里的 0.6/0.4 就是 p(k_t | k_{t-1}) 的具体实现
        
        代码实现对应关系：
            permute_forward:  k_{t-1} - 1（前向索引，对应0.6概率）
            permute_backward: k_{t-1} + 1（后向索引，对应0.4概率）
            change_probs = scatter_v[..., forward] * 0.6 + scatter_v[..., backward] * 0.4
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
        
        根据当前状态计算每个模型的概率。
        
        0.6/0.4 非对称切换概率：
            与 forward 方法一致，使用非对称概率计算模式切换：
            - 前向切换（permute_forward）权重 0.6
            - 后向切换（permute_backward）权重 0.4
            
            数学含义：P(k_t = k ± 1 | k_{t-1}) 的非对称分配
            对应论文公式 20f 中的跳变概率设计
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

        # 定义神经网络层（类LSTM结构，用于学习模型切换模式）
        #
        # 整体架构：
        #   输入(8维) → [遗忘门/缩放门/输入转换] → 隐藏状态(32维) → 输出层 → 输出(8维)
        #                ↑         ↑         ↑
        #           forget    scale   to_reccurrent
        #
        # 五个模块协同工作：
        #   1. forget: 控制历史信息保留程度
        #   2. self_forget: 历史信息自调节（本代码特有）
        #   3. scale: 控制新输入影响程度
        #   4. to_reccurrent: 输入特征转换
        #   5. output_layer: 预测下一时刻模型概率

        # 遗忘门：决定保留多少历史信息（类似LSTM的遗忘门）
        # 结构: Linear(n_models→recurrent_length) + Sigmoid
        # 输入: 当前模型概率分布 (8维)
        # 输出: 遗忘系数 (32维, 0~1)，接近1表示保留，接近0表示遗忘
        # 示例: 输入[0.1,0.1,0.1,0.5,0.1,0.1,0.1,0.1] → 输出[0.62,0.43,0.69,...]
        self.forget = pt.nn.Sequential(
            pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid()
        )

        # 自遗忘门：循环自连接（对历史信息的自调节）
        # 结构: Linear(recurrent_length→recurrent_length) + Sigmoid
        # 输入: 历史隐藏状态 (32维)
        # 输出: 自调节系数 (32维, 0~1)
        # 功能: 让历史信息自我衰减或增强（类似人记忆的自我强化/淡化）
        # 特点: 本代码特有，LSTM中没有这个门
        self.self_forget = pt.nn.Sequential(
            pt.nn.Linear(recurrent_length, recurrent_length), pt.nn.Sigmoid()
        )

        # 缩放门：对输入进行缩放（类似LSTM的输入门）
        # 结构: Linear(n_models→recurrent_length) + Sigmoid
        # 输入: 当前模型概率分布 (8维)
        # 输出: 缩放系数 (32维, 0~1)
        # 功能: 控制新输入有多少被加入到隐藏状态
        # 与to_reccurrent区别: scale输出系数(0~1)，to_reccurrent输出特征(-1~1)
        #
        # 注意: 该模块在当前 forward() 中未被使用（2026-06-30 发现）
        # 可能原因: 代码遗漏/历史遗留/预留扩展
        # 若按标准LSTM设计，应在状态更新时写:
        #   c += self.scale(one_hot) * self.to_reccurrent(one_hot)
        # 当前代码缺少这道"输入门"，新信息直接注入，未经过流量控制
        self.scale = pt.nn.Sequential(
            pt.nn.Linear(n_models, recurrent_length), pt.nn.Sigmoid()
        )

        # 输入转换：将输入映射到循环层
        # 结构: Linear(n_models→recurrent_length) + Tanh
        # 输入: 当前模型概率分布 (8维)
        # 输出: 转换后的特征向量 (32维, -1~1)
        # 功能: 将输入转换到与隐藏状态相同的空间，便于融合
        self.to_reccurrent = pt.nn.Sequential(
            pt.nn.Linear(n_models, recurrent_length), pt.nn.Tanh()
        )

        # 输出层：从循环层输出模型概率
        # 结构: Linear(32→32) + Tanh + Linear(32→8)
        # 输入: 隐藏状态 (32维)
        # 输出: 8个模型的概率logits (8维，未归一化)
        # 后续: 经过softmax得到最终概率分布
        self.output_layer = pt.nn.Sequential(
            pt.nn.Linear(recurrent_length, recurrent_length),
            pt.nn.Tanh(),
            pt.nn.Linear(recurrent_length, n_models),
        )

        # 动态类型: "Boot"(自助法), "Uni"(均匀), "Deter"(确定性)
        self.dyn = dyn
        # 软切换参数: 控制切换的平滑程度
        self.softness = softness

    def init_state(self, batches, n_samples):
        """
        初始化NN切换模型的状态

        返回初始状态，包含模型索引和循环层隐藏状态。
        与Polya_Switching的区别：NN使用循环层状态，Polya使用模型计数器。

        返回：
        ----------
        Tensor: 初始状态
            - r_length > 0: 形状 (batches, n_samples, 1+r_length)
              第0维: 模型索引，后面r_length维: 循环层状态(初始全0)
            - r_length = 0: 形状 (batches, n_samples, 1)
              只有模型索引

        详细说明：
        ----------
        初始化流程:
        1. 设置均匀概率分布: probs = [1/8, 1/8, ..., 1/8]
        2. 根据概率随机采样模型索引
        3. 如果有循环层(r_length>0)，拼接初始为0的循环状态

        示例(batches=2, n_samples=96, r_length=32):
            probs = [0.125, ..., 0.125]  # 8个模型均匀分布

            i_models = pt.multinomial(probs, 192, True)
                     → [3,7,1,0,4,2,5,6,3,1,...]  # 192个随机索引
                     → reshape → 形状(2, 96, 1)
                     → [[[3],[7],[1],...],   # 批次0
                        [[0],[4],[2],...]]   # 批次1

            返回: pt.concat(i_models, zeros(2,96,32), dim=2)
                 → 形状(2, 96, 33)
                 → [[[3, 0,0,...,0],    # 粒子0: 模型3 + 32个0
                     [7, 0,0,...,0],    # 粒子1: 模型7 + 32个0
                     ...],
                    ...]

        与Polya_Switching对比:
            NN_Switching:  [模型索引, 循环层状态(32维)]  → 神经网络学习
            Polya_Switching: [模型索引, 计数器(8维)]   → Polya urn机制
        """
        # 初始化概率分布为均匀分布（各模型概率相等）
        # probs和true_probs都初始化为[1/n_models, ..., 1/n_models]
        # 示例(n_models=8): [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
        self.probs = pt.ones(self.n_models) / self.n_models
        self.true_probs = pt.ones(self.n_models) / self.n_models

        # 根据概率分布随机采样模型索引
        # pt.multinomial(probs, batches*n_samples, True):
        #  multinomial 多项分布   假设 probs = [0.1, 0.6, 0.3] ，表示三个选项的权重：
        # 选项0: 10% 概率被选中
        # 选项1: 60% 概率被选中  ← 最容易被抽到
        # 选项2: 30% 概率被选中 

        #   - 从n_models个模型中，按probs概率，采样batches*n_samples次
        #   - 返回: [3,7,1,0,4,2,5,6,3,1,...] 一维张量
        # .reshape((batches, n_samples, 1)): 重塑为三维，最后一维是模型索引
        # .to(device): 移到指定设备(CPU/GPU)
        i_models = (
            pt.multinomial(self.probs, batches * n_samples, True)
            .reshape((batches, n_samples, 1))
            .to(device=self.device)
        )

        # 如果有循环层(r_length>0)，拼接循环状态
        if self.r_length > 0:
            # 拼接模型索引和初始为0的循环层状态
            # i_models: 形状(batches, n_samples, 1)
            # zeros:    形状(batches, n_samples, r_length)，初始全0
            # 返回:     形状(batches, n_samples, 1+r_length)
            return pt.concat(
                (
                    i_models,
                    pt.zeros((batches, n_samples, self.r_length), device=self.device),
                ),
                dim=2,
            )
        else:
            # 无循环层，只返回模型索引
            # 形状: (batches, n_samples, 1)
            return i_models

    def forward(self, x_t_1, t):
        """
        前向传播：RNN单元

        说明：
        - 使用one-hot编码当前模型
        - 通过遗忘门和输入转换更新循环状态
        - 输出模型概率分布
        - 使用软化参数进行重要性采样校正

        输入: x_t_1 形状 (batches, n_samples, 1+r_length)
              第0维: 模型索引，后面r_length维: 循环层状态
        输出: 形状 (batches, n_samples, 1+r_length)
              第0维: 新模型索引，后面r_length维: 新循环状态
        """
        # 取出当前模型索引，转成整数，增加维度
        # x_t_1[:,:,0]: 形状(batches, n_samples) → unsqueeze → (batches, n_samples, 1)
        # 示例: [[[3],[7],[1],...], [[0],[4],[2],...]]
        old_model = x_t_1[:, :, 0].to(int).unsqueeze(2)

        # One-hot编码：将模型索引转换为one-hot向量
        # 为什么要one-hot？神经网络需要向量输入，而不是单个数字
        # 示例: 模型3 → [0,0,0,1,0,0,0,0]，位置3为1，其他为0
        #
        # 第1步: 创建全0张量，形状(batches, n_samples, n_models)
        # 示例(batches=2, n_samples=3, n_models=8):
        #   [[[0,0,0,0,0,0,0,0],   # 粒子0，8个0
        #     [0,0,0,0,0,0,0,0],   # 粒子1
        #     [0,0,0,0,0,0,0,0]],  # 粒子2
        #    [[0,0,0,0,0,0,0,0],   # 粒子0
        #     [0,0,0,0,0,0,0,0],   # 粒子1
        #     [0,0,0,0,0,0,0,0]]]  # 粒子2
        one_hot = pt.zeros(
            (old_model.size(0), old_model.size(1), self.n_models), device=self.device
        )
        # 第2步: 根据old_model索引，在第2维对应位置填1
        # pt.scatter(one_hot, 2, old_model, 1):
        #   参数2: 在第2维（模型维度）填入
        #   old_model: 索引（在哪个位置填1）
        #   1: 填入的值
        # 示例: old_model=[[[3]],[[7]],[[1]]] → 位置3/7/1填1
        #   [[[0,0,0,1,0,0,0,0],   # 粒子0: 索引3 → 位置3填1
        #     [0,0,0,0,0,0,0,1],   # 粒子1: 索引7 → 位置7填1
        #     [0,1,0,0,0,0,0,0]],  # 粒子2: 索引1 → 位置1填1
        #    ...]
        one_hot = pt.scatter(one_hot, 2, old_model, 1)

        # 获取历史信息（循环层状态）
        # x_t_1[:,:,1:]: 形状(batches, n_samples, r_length)，如(100,200,32)
        # 包含过去时刻的隐藏状态，编码了历史模型切换模式
        old_recurrent = x_t_1[:, :, 1:]

        # RNN更新：类LSTM的状态更新公式
        # 新状态 = 旧状态 ⊙ 自遗忘门 ⊙ 遗忘门 + 输入转换
        #
        # 第1步: old_recurrent * self_forget(old_recurrent)
        #   历史信息自调节：让历史状态自我衰减或增强
        #   示例: [0.5,-0.3,0.8,...] * [0.6,0.7,0.5,...] → [0.30,-0.21,0.40,...]
        #   类比: 人的记忆，重要的自我强化，不重要的逐渐淡化
        #
        # 第2步: * self.forget(one_hot)
        #   根据当前输入决定遗忘多少
        #   示例: [0.30,-0.21,0.40,...] * [0.7,0.8,0.6,...] → [0.21,-0.17,0.24,...]
        #
        # 第3步: + self.to_reccurrent(one_hot)
        #   加入新输入（当前模型的one-hot编码转换后的特征）
        #   示例: [0.21,-0.17,0.24,...] + [0.3,-0.2,0.5,...] → [0.51,-0.37,0.74,...]
        #
        # 与标准RNN的区别:
        #   标准RNN: 新状态 = tanh(W*输入 + U*旧状态)
        #   本代码:  新状态 = 旧状态⊙自遗忘⊙遗忘 + 输入转换
        #   特点: 双遗忘门（forget + self_forget），更精细的控制
        c = old_recurrent * self.self_forget(old_recurrent)
        c *= self.forget(one_hot)
        c += self.to_reccurrent(one_hot)

        # 计算输出概率
        #
        # 第1步: self.output_layer(c)
        #   神经网络输出logits，结构: Linear(32→32)+Tanh+Linear(32→8)
        #   示例: [0.3,-0.2,0.5,0.1,-0.4,0.6,0.2,-0.1]
        #
        # 第2步: pt.abs(...)
        #   取绝对值确保非负（概率不能是负数）
        #   示例: [0.3,0.2,0.5,0.1,0.4,0.6,0.2,0.1]
        #
        # 第3步: / pt.sum(...)
        #   归一化，让所有值加起来=1，变成真正的概率
        #   示例: 总和=2.4 → [0.125,0.083,0.208,0.042,0.167,0.250,0.083,0.042]
        #   模型5概率最高(0.25)，最可能被选中
        probs = pt.abs(self.output_layer(c))
        self.true_probs = probs / pt.sum(probs, dim=2, keepdim=True)

        # 软化处理：混合"网络预测"和"均匀随机"，增加探索能力
        # 公式: correction = softness × true_probs + (1-softness) × 均匀分布
        #
        # softness=1.0: 完全相信网络（100%用true_probs）
        # softness=0.0: 完全随机（100%均匀分布）
        # softness=0.9: 90%网络预测 + 10%随机探索
        #
        # 为什么要软化？类比去餐厅：
        #   纯利用(softness=1): 只点喜欢的菜 → 可能错过其他好菜
        #   纯探索(softness=0): 完全随机点菜 → 可能点到不好吃的
        #   混合(softness=0.9): 90%喜欢的+10%新菜 → 平衡稳定性和新鲜感
        #
        # 示例(softness=0.9, n_models=8):
        #   true_probs = [0.125,0.083,0.208,0.042,0.167,0.250,0.083,0.042]
        #   均匀分布 = [0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125]
        #   correction = 0.9×true_probs + 0.1×均匀分布
        #              = [0.125,0.087,0.200,0.050,0.163,0.238,0.087,0.050]
        #   效果: 高概率略降，低概率略升，让低概率模型也有机会被选中
        #
        # .detach()的作用: 切断梯度传播
        #   因为后面的采样操作(pt.multinomial)不可导
        #   如果不断开，反向传播时会出错
        #   类比: 学生答题→老师批改→到这里为止→后面的实际应用不影响学习过程
        self.correction = (
            self.softness * self.true_probs.detach()
            + (1 - self.softness) / self.n_models
        )
        probs = self.correction

        # 根据软化后的概率采样新模型，拼接新状态返回
        #
        # 第1步: probs.reshape(-1, self.n_models)
        #   展平为(batches*n_samples, n_models)，如(20000, 8)
        #
        # 第2步: pt.multinomial(..., 1, True)
        #   每行采样1个模型，返回新模型索引
        #   示例: [[3],[7],[1],...] 形状(20000, 1)
        #
        # 第3步: .reshape([batches, n_samples, 1])
        #   重塑为(batches, n_samples, 1)，如(100, 200, 1)
        #
        # 第4步: pt.concat((新模型, c), dim=2)
        #   拼接新模型索引和新循环状态
        #   返回形状: (batches, n_samples, 1+r_length)，如(100, 200, 33)
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

        重要性权重 = log(真实概率 / 提议概率)
        - 权重>0: 真实概率>提议概率，粒子权重增加（模型被低估）
        - 权重<0: 真实概率<提议概率，粒子权重减少（模型被高估）

        示例:
            真实概率=0.25, 提议概率=0.20 → log(0.25/0.20) = 0.22 → 权重增加
            真实概率=0.15, 提议概率=0.20 → log(0.15/0.20) = -0.29 → 权重减少
        """
        # 取出当前时刻实际选择的模型索引
        # x_t[:,:,0]: 形状(batches, n_samples)
        # 示例: [[3,7,1,...], [0,4,2,...]]
        models = x_t[:, :, 0].to(int)

        # 找出每个粒子实际选择模型的真实概率
        #
        # 第1步: self.true_probs.reshape(-1, n_models)
        #   展平为(batches*n_samples, n_models)，如(20000, 8)
        #
        # 第2步: models.flatten()
        #   展平模型索引为一维，如[3,7,1,0,4,2,...]
        #
        # 第3步: batched_select(true_probs, models)
        #   根据索引从true_probs中选择对应概率
        #   示例: true_probs=[[0.125,0.083,0.208,0.042,0.167,0.250,0.083,0.042]]
        #         models=5 → 选择0.250
        #
        # 第4步: .reshape(x_t.size(0), x_t.size(1))
        #   恢复原始形状(batches, n_samples)
        probs = batched_select(
            self.true_probs.reshape(-1, self.n_models), models.flatten()
        ).reshape(x_t.size(0), x_t.size(1))

        # 找出每个粒子实际选择模型的提议概率（软化后的概率）
        # 计算过程同上，只是从self.correction中选择
        #
        # 示例: correction=[[0.125,0.087,0.200,0.050,0.163,0.238,0.087,0.050]]
        #       models=5 → 选择0.238
        corrections = batched_select(
            self.correction.reshape(-1, self.n_models), models.flatten()
        ).reshape(x_t.size(0), x_t.size(1))

        # 计算对数权重: log(真实概率 / 提议概率 + ε)
        # +1e-7: 防止除以0或log(0)
        #
        # 示例: probs=0.250, corrections=0.238
        #       log(0.250/0.238 + 1e-7) = log(1.05) = 0.049
        #
        # 在粒子滤波中的作用:
        #   1. 用提议分布(correction)采样新模型
        #   2. 计算重要性权重校正偏差
        #   3. 权重高的粒子保留，权重低的淘汰
        #   4. 重采样，进入下一时刻
        return pt.log(probs / corrections + 1e-7)

    def get_regime_probs(self, r_t_1):
        """
        获取模型概率

        根据循环状态计算每个模型的对数概率，用于IMM算法的模型混合。
        与forward的区别：不使用软化参数，直接输出原始概率分布。

        输入: r_t_1 形状 (batches, n_samples, r_length)
              循环层隐藏状态，编码了历史模型切换模式
        输出: 形状 (batches, n_samples, n_models)
              每个模型的对数概率

        示例(1个粒子，n_models=8):
            r_t_1 = [0.51, -0.37, 0.74, ...]  # 32维循环状态
            output_layer → [0.3, -0.2, 0.5, 0.1, -0.4, 0.6, 0.2, -0.1]
            pt.abs → [0.3, 0.2, 0.5, 0.1, 0.4, 0.6, 0.2, 0.1]
            归一化 → [0.125, 0.083, 0.208, 0.042, 0.167, 0.250, 0.083, 0.042]
            pt.log → [-2.08, -2.49, -1.57, -3.17, -1.79, -1.39, -2.49, -3.17]
        """
        # 神经网络输出logits，取绝对值确保非负
        # self.output_layer: Linear(32→32)+Tanh+Linear(32→8)
        # +1e-7: 防止全0导致后续归一化除以0
        # 输入: r_t_1 形状(batches, n_samples, r_length)
        # 输出: 形状(batches, n_samples, n_models)
        probs = pt.abs(self.output_layer(r_t_1) + 1e-7)

        # 归一化：让所有模型的概率加起来=1
        # pt.sum(probs, dim=2, keepdim=True): 在第2维求和，保持维度
        # 示例: [0.3,0.2,0.5,0.1,0.4,0.6,0.2,0.1] 总和=2.4
        #       → [0.125,0.083,0.208,0.042,0.167,0.250,0.083,0.042]
        probs = probs / pt.sum(probs, dim=2, keepdim=True)

        # 返回对数概率，用于IMM算法的模型混合
        # 示例: [-2.08, -2.49, -1.57, -3.17, -1.79, -1.39, -2.49, -3.17]
        #       模型5对数概率最高(-1.39)，表示最可能切换到模型5
        return pt.log(probs)

    def R_0(self, batches, n_samples, k):
        """
        初始化IMM辅助变量

        创建全0循环状态，调用R_t用模型k的one-hot编码更新一次，
        返回初始辅助变量（表示"假设当前在模型k"的初始隐藏状态）。

        参数说明：
        ----------
        batches: int
            批量大小
        n_samples: int
            每批样本数（粒子数）
        k: int
            模型索引（当前活跃的模型）

        返回：
        ----------
        Tensor: 初始化的辅助变量，形状 (batches, n_samples, r_length)
                32维循环状态，表示"假设在模型k"的初始隐藏状态

        详细说明：
        ----------
        执行流程:
        1. 创建全0张量: pt.zeros((batches, n_samples, r_length))
           示例(batches=2, n_samples=3, r_length=32):
           [[[0,0,...,0], [0,0,...,0], [0,0,...,0]],
            [[0,0,...,0], [0,0,...,0], [0,0,...,0]]]
        2. 调用 R_t(全0张量, k) 进行一次RNN更新

        与Polya_Switching的R_0对比:
            NN_Switching: 全0循环状态 → R_t更新 → 32维隐藏状态
            Polya_Switching: 全1计数器，模型k设为2 → 8维计数器
        """
        # 创建全0循环状态，形状(batches, n_samples, r_length)
        # 示例: (2, 3, 32) 表示2批，每批3个粒子，32维循环状态
        return self.R_t(
            pt.zeros((batches, n_samples, self.r_length), device=self.device), k
        )

    def R_t(self, r_t_1, k):
        """
        更新IMM辅助变量（RNN前向传播）

        根据模型k更新循环状态，执行一次RNN前向传播。
        这是类LSTM的更新：新状态 = 旧状态⊙自遗忘⊙遗忘 + 输入转换

        参数说明：
        ----------
        r_t_1: Tensor
            上一时刻的循环状态，形状 (batches, n_samples, r_length)
        k: int
            模型索引（当前活跃的模型）

        返回：
        ----------
        Tensor: 更新后的循环状态，形状 (batches, n_samples, r_length)

        详细说明：
        ----------
        执行流程（以k=3为例）:

        Step1: 创建one-hot向量
            zero_vec = [0, 0, 0, 1, 0, 0, 0, 0]  # 模型3位置为1

        Step2: 历史信息自调节
            c = r_t_1 * self.self_forget(r_t_1)
            示例: r_t_1=[0.5,-0.3,0.8,...] * [0.6,0.7,0.5,...]
                  → [0.30,-0.21,0.40,...]

        Step3: 根据当前输入遗忘
            c = c * self.forget(zero_vec)
            示例: [0.30,-0.21,0.40,...] * [0.7,0.8,0.6,...]
                  → [0.21,-0.17,0.24,...]

        Step4: 加入新输入
            c = c + self.to_reccurrent(zero_vec)
            示例: [0.21,-0.17,0.24,...] + [0.3,-0.2,0.5,...]
                  → [0.51,-0.37,0.74,...]

        在IMM算法中的作用:
            R_0(k): 初始化辅助变量（"假设在模型k"的初始状态）
            R_t(r,k): 更新辅助变量（RNN前向传播）
            这两个方法让神经网络能为IMM提供模型混合概率

        与Polya_Switching的R_t对比:
            NN_Switching: RNN前向传播，返回32维隐藏状态
            Polya_Switching: 计数器加1，返回8维计数器
        """
        # 创建模型k的one-hot编码
        # zero_vec: [0,0,...,1,...,0]，k位置为1，其他为0
        self.zero_vec = pt.zeros(self.n_models, device=self.device)
        self.zero_vec[k] = 1

        # RNN更新：新状态 = 旧状态⊙自遗忘⊙遗忘 + 输入转换
        # Step1: 历史信息自调节（让历史状态自我衰减或增强）
        c = r_t_1 * self.self_forget(r_t_1)
        # Step2: 根据当前输入决定遗忘多少
        c = c * self.forget(self.zero_vec)
        # Step3: 加入新输入（当前模型的one-hot编码转换后的特征）
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
        """
        动态构建多层神经网络

        构建结构: 输入层 + (layers-2)个隐藏层 + 输出层
        每层使用Tanh激活函数，输出层无激活函数

        参数说明:
        ----------
        input: int
            输入维度
        hidden: int
            隐藏层维度
        output: int
            输出维度
        layers: int
            总层数（至少2层）

        示例(layers=3, input=4, hidden=32, output=4):
            构建过程:
            1. [Linear(4→32), Tanh]                    # 输入层
            2. range(1) → 添加 [Linear(32→32), Tanh]   # 1个隐藏层
            3. [Linear(32→4)]                          # 输出层

            最终网络:
            Sequential(
                Linear(4, 32), Tanh,      # 输入层
                Linear(32, 32), Tanh,     # 隐藏层
                Linear(32, 4)             # 输出层
            )
            数据流: 4 → 32 → 32 → 4

        示例(layers=5):
            数据流: 4 → 32 → 32 → 32 → 32 → 4
            (输入层 + 3个隐藏层 + 输出层)
        """
        super().__init__()

        # 第1步: 创建输入层 + Tanh激活
        # Linear(input→hidden): 将输入维度映射到隐藏维度
        # Tanh: 激活函数，输出范围(-1, 1)
        nn_layers = [pt.nn.Linear(input, hidden), pt.nn.Tanh()]

        # 第2步: 循环创建(layers-2)个隐藏层
        # 每个隐藏层: Linear(hidden→hidden) + Tanh
        # 示例: layers=3 → range(1) → 1个隐藏层
        #       layers=5 → range(3) → 3个隐藏层
        for i in range(layers - 2):
            nn_layers += [pt.nn.Linear(hidden, hidden), pt.nn.Tanh()]

        # 第3步: 创建输出层
        # Linear(hidden→output): 将隐藏维度映射到输出维度
        # 注意: 输出层没有激活函数（回归任务通常不需要）
        nn_layers += [pt.nn.Linear(hidden, output)]

        # 第4步: 组装成Sequential网络
        # =====================================================================
        # self.net = pt.nn.Sequential(*tuple(nn_layers))
        # 
        # 代码拆解：
        #   nn_layers        → 列表，包含多个网络层（如 Linear、ReLU、Tanh）
        #   tuple(nn_layers) → 把列表转成元组
        #   *                → Python 解包操作符，把元组展开成多个参数
        #   pt.nn.Sequential → PyTorch 的顺序容器，按顺序堆叠各层
        #   self.net         → 把构建好的网络保存为实例属性
        # 
        # 示例：假设 nn_layers = [Linear(10,20), Tanh(), Linear(20,5)]
        #       则这行代码等价于：
        #       self.net = pt.nn.Sequential(
        #           Linear(10, 20),
        #           Tanh(),
        #           Linear(20, 5)
        #       )
        # 
        # 通俗比喻：
        #   nn_layers = 一堆乐高积木（每块是一个网络层）
        #   *tuple(...) = 把积木从盒子里拿出来，一块一块摆好
        #   Sequential = 用胶水按顺序粘起来，变成一个完整的模型
        #   self.net = 把粘好的模型存起来，随时可以调用
        # 
        # 为什么这样写？
        #   nn_layers 的内容是动态生成的（根据参数决定有几层、每层多大）
        #   先收集到列表里，最后一次性用 Sequential 组装成完整网络
        # =====================================================================
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

    学习导航：
    第一阶段先阅读项目根目录的《PF基础学习笔记.md》，再回到本类。
    本阶段只要求讲清 3 个方法：
    - M_0_proposal(): 生成 t=0 的初始粒子
    - M_t_proposal(): 将粒子从 t-1 推进到 t
    - log_f_t(): 用观测 y_t 给当前粒子打分

    用于生成模拟数据的标准粒子滤波
    状态转移方程：x_t = a[k] * x_{t-1} + b[k] + noise
    观测方程：y_t = a[k] * sqrt(|x_t|) + b[k] + noise

    继承关系:
    ----------
    Feynman_Kac (model.py)
        ↓ 继承
    SSM (model.py)
        ↓ 继承
    HMM (model.py)
        ↓ 继承
    PF (本类，Net.py) ← 具体实现

    继承的方法:
    ----------
    从 Feynman_Kac 继承:
        - set_observations(): 抽象接口，本类必须实现
        - log_G_0(), log_G_t(): 权重函数
        - M_0_proposal(), M_t_proposal(): 提议分布采样

    从 SSM 继承:
        - log_R_0(), log_R_t(): Radon-Nikodym导数
        - log_f_t(): 观测似然
        - log_eta_t(): 辅助权重

    从 HMM 继承:
        - M_0_proposal(), M_t_proposal(): 具体采样实现
        - generate_state_0(), generate_state_t(): 状态生成

    本类实现的方法:
    ----------
    - set_observations(): 使用reindexed_array存储观测值
    - __init__(): 初始化模型参数
    - 其他PF特有的方法

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

    使用示例:
    ----------
    # 创建PF实例
    pf = PF(a=[1.0, 0.5], b=[0.0, 0.5], var_s=0.1, switching_dyn=markov_switch)

    # 设置观测值（调用继承自Feynman_Kac的接口，本类实现）
    pf.set_observations(data.get_observation, t=5)
    # 内部使用reindexed_array存储y_4和y_5

    # 进行粒子滤波
    pf.M_0_proposal(batches=10, n_samples=100)  # 从HMM继承的具体实现
    """

    def set_observations(self, get_observation: Callable, t: int):
        """
        设置观测值（实现Feynman_Kac基类的抽象接口）

        继承自model.py中的Feynman_Kac基类，子类必须实现此方法。
        使用reindexed_array存储t-1和t两个时刻的观测值，支持逻辑索引访问。

        参数说明：
        ----------
        get_observation: Callable
            获取观测值的函数，来自model.py中的_get_observation方法
            调用方式: get_observation(t) → 返回时间t的观测值
        t: int
            当前时间步（逻辑索引）

        详细说明：
        ----------
        执行流程（以t=5为例）:
        1. get_observation(4) → 获取y_4（上一时刻观测）
        2. get_observation(5) → 获取y_5（当前时刻观测）
        3. 创建reindexed_array(base_index=4, array=[y4, y5])
        4. 赋值给self.y

        reindexed_array的作用:
        - 实际存储: array=[y4, y5]，实际索引0和1
        - 逻辑索引: base_index=4，表示array[0]对应逻辑索引4
        - 访问方式: self.y[4]→y4, self.y[5]→y5（自动转换）

        为什么要用逻辑索引:
        - 代码直观: 用self.y[t]直接访问时间t的观测
        - 隐藏细节: 不需要关心数组实际从0开始存储
        - 减少错误: 避免手动计算t-start_index的偏移

        继承关系:
        Feynman_Kac(基类, model.py)
            ↓
        SSM
            ↓
        PF/IMMPF/...(子类, Net.py) ← 在此实现具体逻辑

        与基类的区别:
        - Feynman_Kac: 定义抽象接口，抛出NotImplementedError
        - PF/IMMPF: 具体实现，使用reindexed_array存储观测
        """
        # 获取t-1和t两个时刻的观测值，用reindexed_array存储
        # base_index=t-1: 第一个元素(y_{t-1})的逻辑索引是t-1
        # [get_observation(t-1), get_observation(t)]: 实际存储的数组
        #
        # 示例(t=5):
        #   get_observation(4) → y_4 = [1.2, 0.8, -0.3, 0.5]
        #   get_observation(5) → y_5 = [1.3, 0.9, -0.2, 0.6]
        #   self.y = reindexed_array(base_index=4, array=[y4, y5])
        #
        # 后续使用:
        #   self.y[4] → y_4（实际array[4-4]=array[0]）
        #   self.y[5] → y_5（实际array[5-4]=array[1]）
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
        """
        初始化多模型粒子滤波器（可微分交互多模型粒子滤波器 - Differentiable IMM-PF）

        这是一个多模型粒子滤波器的初始化方法，继承自 SSM 类。支持可学习的模式切换
        和端到端训练，适用于具有模式切换的动态系统（如机动目标跟踪、故障诊断等）。

        参数详解：
        ----------
        a: List[int]
            各模型的参数 a（模型标识或模式参数）。
            示例: a = [0, 1, 2] 表示 3 个模型
            在状态转移中使用: x_t = a[k] * x_{t-1} + b[k] + noise

        b: List[int]
            各模型的参数 b（与 a 配合使用）。
            与 a 一一对应，共同定义各模型的动态特性。

        var_s: float
            状态噪声方差 σ²。
            用于定义状态转移噪声和观测噪声的分布。
            影响粒子扩散程度和估计精度。

        switching_dyn: pt.nn.Module
            模式切换动态（神经网络）。
            用于学习模式转移概率 Π_ij = P(模式 j | 模式 i)。
            与传统 IMM 的固定转移矩阵不同，此处使用可学习的神经网络，
            支持端到端训练和自动优化切换策略。

        dyn: str, default="Boot"
            滤波器类型选择。
            - "Boot": Bootstrap 粒子滤波（提议=先验，简单高效）
            - 其他: Guided 粒子滤波（使用观测信息引导提议，适合复杂问题）

        device: str, default="cuda"
            计算设备（"cuda" 或 "cpu"）。

        初始化流程：
        -----------
        1. 设置模型参数 (a, b, n_models)
           - n_models = len(a): 模型/模式的数量
           - 支持多模型 IMM 结构

        2. 设置模式切换网络 (switching_dyn)
           - 可学习的模式转移概率
           - 与传统 IMM 的固定矩阵不同，支持梯度优化

        3. 定义概率分布
           - x_dist: 状态转移噪声 N(0, σ²)
           - init_x_dist: 初始状态分布 U(-0.5, 0.5)
           - y_dist: 观测噪声 N(0, σ²)

        4. 预计算优化 (var_factor)
           - var_factor = -1 / (2 * var_s)
           - 用于加速高斯对数似然计算

        5. 选择滤波算法
           - Bootstrap: 简单通用，提议分布等于先验
           - Guided: 高效但需要额外实现引导函数

        与 model.py 的关联：
        -------------------
        - model.py (SSM 类): 定义抽象接口和辅助粒子滤波框架
        - Net.py (本类): 具体实现多模型 IMM-PF，支持可微分训练

        关键区别：
        - SSM (model.py): 单模型框架，抽象层级
        - PF/IMMPF (Net.py): 多模型实现，完全可微分，使用 reindexed_array 存储观测

        使用场景：
        ---------
        - 目标跟踪（机动目标）: 不同运动模式（匀速、加速、转弯）
        - 故障诊断: 不同故障模式切换
        - 金融时序: 市场状态切换（牛市/熊市）
        - 任何具有模式切换的动态系统

        "可微分"的意义：
        ---------------
        - 整个滤波过程可以反向传播梯度
        - 可以与神经网络端到端联合训练
        - switching_dyn 可以自动学习最优模式切换策略
        """
        super().__init__(device)

        # 1. 模型数量与参数存储
        # n_models: 告诉系统有多少个并行的动态模型（IMM 的核心）
        # a, b: 存储在 GPU/CPU 上，供后续计算使用
        # 示例: a = [0, 1, 2], b = [1, 2, 3] → n_models = 3（三模型 IMM）
        self.n_models = len(a)
        self.a = pt.tensor(a, device=device)
        self.b = pt.tensor(b, device=device)

        # 2. 模式切换动态网络
        # 存储模式切换概率的神经网络
        # 在 IMM-PF 中，用于计算模式转移矩阵 Π_ij = P(模式 j | 模式 i)
        # 与传统 IMM 的区别: 固定转移矩阵 → 神经网络学习（可端到端训练）
        self.switching_dyn = switching_dyn

        # 3. 状态分布定义
        # x_dist: 状态转移噪声 N(0, σ²)，用于状态转移时的随机扰动
        # init_x_dist: 初始状态分布 U(-0.5, 0.5)，表示对初始状态的完全不确定性

        # x_dist: 状态转移噪声分布 N(0, σ²)
        # 用于状态传播时添加过程噪声，模拟系统动态的不确定性
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        # init_x_dist: 初始状态分布 U(-0.5, 0.5)
        # 用于t=0时刻初始化粒子，表示对系统初始状态的完全不确定性（先验知识很少）
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)

        # 4. 方差因子预计算（优化）
        # var_factor = -1 / (2 * σ²)
        # 用于快速计算高斯分布的对数似然指数部分
        # 高斯 PDF: p(x) ∝ exp(-x² / (2σ²)) = exp(var_factor * x²)
        self.var_factor = -1 / (2 * var_s)

        # 5. 观测噪声分布
        # 定义观测噪声 N(0, σ²)，用于 log_f_t（观测似然）的计算
        self.y_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))

        # 6. 滤波器类型选择
        # Bootstrap: 提议 = 先验，简单高效，适用于标准问题
        # Guided: 使用观测信息引导提议，适用于复杂/高维问题
        if dyn == "Boot":
            self.alg = self.PF_Type.Bootstrap
        else:
            self.alg = self.PF_Type.Guided

    def M_0_proposal(self, batches: int, n_samples: int):
        """
        时间0的提议分布

        这是粒子滤波初始化阶段的核心方法，用于生成初始粒子集合。
        在 IMM-PF（交互多模型粒子滤波）中，每个粒子同时包含位置和模式信息。

        参数说明：
        ----------
        batches: int
            批量大小，表示同时处理的独立序列数量
        n_samples: int
            每批样本数，表示每个序列的粒子数量

        返回：
        ----------
        Tensor: 初始状态，形状 (batches, n_samples, state_dim)
                包含位置（第0维）和模型/模式索引（后续维度）

        详细解释：
        -----------
        本方法实现了时间 0 的提议分布采样，是粒子滤波的第一步。

        与传统粒子滤波的区别：
        - 标准 PF: 粒子只有状态 x = [position]
        - IMM-PF: 粒子有状态 + 模式 x = [position, regime]
                                   ↑            ↑
                                位置(连续)   模式(离散/连续)

        代码逻辑详解：
        --------------

        1. 采样初始位置 (init_locs):
           - 从 init_x_dist = Uniform(-0.5, 0.5) 采样
           - 表示对初始位置的完全不确定性
           - 形状变化:
             * sample([batches, n_samples]) → (batches, n_samples)
             * .to(device) → 移动到 GPU/CPU
             * .unsqueeze(2) → (batches, n_samples, 1)

        2. 采样初始模式 (init_regimes):
           - 调用 switching_dyn.init_state(batches, n_samples)
           - 生成每个粒子的初始模式/模型索引
           - 这是 IMM (Interacting Multiple Model) 的核心特性
           - 形状: (batches, n_samples, n_regimes)

        3. 组合状态:
           - pt.cat((init_locs, init_regimes), dim=2)
           - 在第2维（最后一维）拼接位置和模式
           - 最终形状: (batches, n_samples, 1 + n_regimes)

        最终状态结构：
        粒子状态 x_0 = [loc, regime_0, regime_1, ..., regime_{n-1}]
                       ↑    ↑
                    位置   模式指示器（哪个模型激活）

        与 model.py 的关联：
        -------------------
        - model.py (Feynman_Kac/SSM): M_0_proposal 是抽象方法
        - Net.py (本实现): 提供具体实现，支持多模型 IMM 特性

        关键设计要点：
        --------------
        1. 批量处理：同时处理多个独立序列（batches 维度）
        2. IMM 特性：每个粒子携带模式信息，支持多模型切换
        3. 设备管理：显式调用 .to(device) 确保张量在正确设备上
        4. 维度对齐：使用 unsqueeze 确保拼接维度一致
        5. 可微分性：整个采样过程可反向传播（如果 switching_dyn 可微）

        使用示例：
        ----------
        >>> batches = 4       # 4 个独立序列
        >>> n_samples = 100   # 每个序列 100 个粒子
        >>> x_0 = model.M_0_proposal(batches, n_samples)
        >>> x_0.shape
        torch.Size([4, 100, 4])  # state_dim = 1 + 3

        >>> positions = x_0[:, :, 0]      # 所有粒子的位置
        >>> regimes = x_0[:, :, 1:]       # 所有粒子的模式指示
        """
        # 1. 采样初始位置
        # 从 Uniform(-0.5, 0.5) 采样，表示对初始位置的完全不确定性
        # 形状变化: (batches, n_samples) -> (batches, n_samples, 1)
        init_locs = (
            self.init_x_dist.sample([batches, n_samples])
            .to(device=self.device)
            .unsqueeze(2)
        )

        # 2. 采样初始模式/模型
        # 调用模式切换网络的 init_state 方法
        # 生成每个粒子的初始模式索引（IMM 核心特性）
        # 形状: (batches, n_samples, n_regimes)
        init_regimes = self.switching_dyn.init_state(batches, n_samples)

        # 3. 组合位置和模式
        # 在第2维拼接，形成完整的状态向量
        # 最终形状: (batches, n_samples, 1 + n_regimes)
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
        # 1. 生成过程噪声：从 N(0, σ²) 采样，形状为 (batches, n_samples)
        # 为每个粒子添加独立的过程噪声，模拟系统动态的不确定性
        noise = self.x_dist.sample([x_t_1.size(0), x_t_1.size(1)]).to(
            device=self.device
        )
        
        # 2. 模式切换：使用神经网络决定粒子切换到哪个动态模型
        # x_t_1[:, :, 1:] 提取状态中的模式信息（去掉位置部分）
        # new_models 包含新的模型概率/索引，形状为 (batches, n_samples, n_models)
        new_models = self.switching_dyn(x_t_1[:, :, 1:], t)
        
        # 3. 提取模型索引：从 new_models 中获取当前选定的模型索引
        # index 形状为 (batches, n_samples)，用于后续查表获取模型参数
        index = new_models[:, :, 0].to(int)
        
        # 4. 查表获取模型参数：根据模型索引从 a 和 b 数组中查找对应参数
        # scaling = a[k]：状态转移的斜率参数
        # bias = b[k]：状态转移的偏置参数
        scaling = self.a[index]
        bias = self.b[index]
        
        # 5. 计算新位置：执行线性状态转移方程 x_t = a·x_{t-1} + b + noise
        # x_t_1[:, :, 0] 是前一时刻的位置
        # .unsqueeze(2) 增加维度以匹配噪声形状，便于广播相加
        new_pos = (scaling * x_t_1[:, :, 0] + bias).unsqueeze(2) + noise
        
        # 6. 组合新状态：将新位置和新模型信息在第2维拼接
        # 返回形状为 (batches, n_samples, state_dim) 的完整状态
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
        时间t的Radon-Nikodym导数（Radon-Nikodym Derivative）
        
        在粒子滤波中，R-N导数用于计算重要性权重，修正提议分布与目标分布之间的差异。
        
        数学意义：
            R_t = dQ/dP，其中 Q 是目标分布（后验），P 是提议分布
        
        参数说明：
        ----------
        x_t: Tensor
            当前时刻状态，形状 (batches, n_samples, state_dim)
        x_t_1: Tensor
            前一时刻状态，形状 (batches, n_samples, state_dim)
        t: int
            当前时间步
        
        返回：
        ----------
        Tensor: 模型切换的对数概率，形状 (batches, n_samples)
        
        说明：
        - Bootstrap滤波：提议分布 = 先验分布，此时 R_t = 1（无需修正）
        - Guided滤波：提议分布 ≠ 先验分布，需要用 R_t 进行权重修正
        - 本方法计算从模型 m_{t-1} 切换到 m_t 的对数转移概率：log P(m_t | m_{t-1})
        
        权重更新公式：
            w_t ∝ w_{t-1} * R_t * f_t
            其中 f_t 是观测似然（由 log_f_t 计算）
        """
        # x_t[:, :, 1:] 和 x_t_1[:, :, 1:] 提取状态中的模式信息（去掉位置）
        # 计算模型切换的对数概率，用于重要性采样权重修正
        return self.switching_dyn.get_log_probs(x_t[:, :, 1:], x_t_1[:, :, 1:])

    def log_f_t(self, x_t, t: int):
        """
        观测似然（Observation Likelihood）
        
        计算给定状态下观测值的对数似然 log p(y_t | x_t)，用于粒子滤波中的权重更新。
        
        参数说明：
        ----------
        x_t: Tensor
            当前状态，形状 (batches, n_samples, state_dim)
            包含粒子位置和模型信息
        t: int
            当前时间步
        
        返回：
        ----------
        Tensor: 观测似然的对数，形状 (batches, n_samples)
            值越大表示该粒子解释当前观测的能力越强
        
        计算流程：
        ----------
        1. 提取模型索引：确定每个粒子使用哪个动态模型
        2. 查表获取参数：根据模型索引获取对应的 a[k] 和 b[k]
        3. 预测观测位置：使用非线性观测方程 locs = a[k] * sqrt(|x_t|) + b[k]
        4. 计算高斯似然：log p(y_t | x_t) ∝ -((y_t - locs)^2) / (2 * var_s)
        
        数学推导：
        ----------
        高斯分布 PDF: p(y|x) = (1/√(2πσ²)) * exp(-(y-locs)²/(2σ²))
        
        取对数: log p(y|x) = -0.5*log(2πσ²) - (y-locs)²/(2σ²)
                          ∝ var_factor * (y - locs)²
        
        其中 var_factor = -1/(2*var_s)
        
        在粒子滤波中的作用：
        ----------
        权重更新: w_t^(i) ∝ w_{t-1}^(i) * p(y_t | x_t^(i))
        
        - 预测位置 locs 与实际观测 y[t] 越接近 → 似然值越大 → 粒子权重越高
        - 预测位置与观测偏差越大 → 似然值越小 → 粒子权重越低
        
        这是重采样的依据，确保高似然粒子被保留，低似然粒子被剔除。
        """
        # 1. 提取模型索引：x_t[:, :, 1] 是每个粒子携带的模型编号
        # index 形状: (batches, n_samples)，用于后续查表
        index = x_t[:, :, 1].to(int)
        
        # 2. 查表获取模型参数：根据模型索引获取对应的斜率和偏置
        # scaling = a[k]: 观测方程的斜率参数
        # bias = b[k]: 观测方程的偏置参数
        scaling = self.a[index]
        bias = self.b[index]
        
        # 3. 计算预测观测位置：使用非线性观测方程
        # x_t[:, :, 0] 是粒子的位置（状态的第一维）
        # pt.abs(x) + 1e-7: 取绝对值并加小常数，防止 sqrt(0)
        # 观测方程: locs = a[k] * sqrt(|x_t|) + b[k]
        locs = scaling * pt.sqrt(pt.abs(x_t[:, :, 0]) + 1e-7) + bias
        
        # 4. 计算高斯似然（对数形式）
        # self.y[t] 是当前时刻的真实观测值
        # (y - locs)^2: 预测误差平方
        # var_factor = -1/(2*var_s): 方差相关的常数因子
        # 返回值形状: (batches, n_samples)，每个粒子的对数似然分数
        return self.var_factor * ((self.y[t] - locs) ** 2)

    def observation_generation(self, x_t):
        """
        从状态生成观测（Observation Generation）
        
        实现观测方程，用于从隐藏状态生成观测值。这是正向生成模型，
        与 log_f_t（逆向推断）共同定义完整的观测模型。
        
        参数说明：
        ----------
        x_t: Tensor
            当前状态，形状 (batches, n_samples, state_dim)
            包含粒子位置和模型信息
        
        返回：
        ----------
        Tensor: 生成的观测值，形状 (batches, n_samples, 1)
        
        观测方程：
        ----------
        y_t = a[k] * sqrt(|x_t|) + b[k] + noise
        
        其中：
        - a[k], b[k]: 模型k的观测参数
        - noise ~ N(0, σ²): 观测噪声
        
        与 log_f_t 的关系：
        ----------------
        这个方法与 log_f_t 是互逆的关系：
        
        | 方法 | 方向 | 功能 |
        |------|------|------|
        | observation_generation | 状态 → 观测 | 生成合成观测数据 |
        | log_f_t | 状态 + 观测 → 似然 | 评估粒子与观测的匹配程度 |
        
        两者共用同一个观测方程，前者用于生成数据，后者用于推断时的似然计算。
        
        应用场景：
        ----------
        1. 模拟数据生成：在训练或测试时生成合成观测数据
        2. 模型验证：验证观测模型的正确性
        3. 理解观测结构：展示状态如何映射到观测空间
        
        数据生成流程：
        ------------
        真实状态 x_t
            │
            ▼
        observation_generation(x_t)
            │
            ▼
        y_t = a·√|x| + b + noise ──► 模拟观测数据
            │
            ▼
        用于训练/测试粒子滤波器
        """
        # 1. 生成观测噪声：从 N(0, σ²) 采样
        # 形状: (batches, 1)，为每个批次添加独立的观测噪声
        noise = self.y_dist.sample((x_t.size(0), 1)).to(device=self.device)
        
        # 2. 提取模型索引：确定每个粒子使用哪个观测模型
        # index 形状: (batches, n_samples)
        index = x_t[:, :, 1].to(int)
        
        # 3. 查表获取模型参数：根据模型索引获取对应的 a[k] 和 b[k]
        # scaling = a[k]: 观测方程的斜率参数
        # bias = b[k]: 观测方程的偏置参数
        scaling = self.a[index]
        bias = self.b[index]
        
        # 4. 计算观测值：使用非线性观测方程
        # x_t[:, :, 0]: 粒子的位置（状态的第一维）
        # pt.sqrt(pt.abs(...)): 平方根变换（非线性观测）
        # .unsqueeze(2): 增加维度以匹配噪声形状，便于广播相加
        # 观测方程: y_t = a[k] * sqrt(|x_t|) + b[k] + noise
        new_pos = (scaling * pt.sqrt(pt.abs(x_t[:, :, 0])) + bias).unsqueeze(2) + noise
        
        # 返回生成的观测值，形状: (batches, n_samples, 1)
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
        # pt.distributions.Normal PyTorch 的正态分布类，用于生成粒子位置
        self.x_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)
        var_s = pt.tensor(var_s)
        self.var_factor = -1 / (2 * var_s)
        self.y_dist = pt.distributions.Normal(pt.zeros(1), sqrt(var_s))
        self.alg = self.PF_Type.Bootstrap

        # =========================================================================
        # 高斯对数似然预计算因子（性能优化）
        # =========================================================================
        # 数学背景：
        #   对于高斯分布 N(x; μ, σ²)，其概率密度函数为：
        #   p(x) = 1/√(2πσ²) * exp(-(x-μ)²/(2σ²))
        #
        #   对数似然：
        #   log p(x) = -1/2*log(2πσ²) - (x-μ)²/(2σ²)
        #            = pre_factor + var_factor * (x-μ)²
        #
        # 代码与数学的对应关系：
        #   var_factor = -1/(2*σ²)          ← 方差相关因子（指数部分系数）
        #   pre_factor = -1/2*log(2πσ²)     ← 归一化常数（对数部分）
        #
        # 为什么要预计算？
        #   在粒子滤波中，需要反复计算每个粒子的观测似然。
        #   不复用：每次计算都重复计算常数部分，效率低下。
        #   预计算：初始化时计算一次，后续只需乘法和加法，大幅提升效率。
        #
        # 数值稳定性保护（+ 1e-6）：
        #   - 防止 var_s = 0 时的除零错误
        #   - 防止 var_s ≈ 0 时的数值爆炸
        #   - 防止 var_s < 0 时对负数取对数得到 nan
        #   - 当 var_s = 0 时，实际使用 var_s = 1e-6，保证计算稳定
        #
        # 使用示例：
        #   log_likelihood = var_factor * (x - mu)**2 + pre_factor
        #   等价于：log N(x; mu, var_s)
        #
        # 在粒子滤波中的应用：
        #   用于计算观测似然 log_f_t，评估粒子与观测的匹配程度
        #   是权重计算的核心组成部分
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

        # 采样初始位置
        # init_x_dist = Uniform(-0.5, 0.5)，表示对初始位置的完全不确定性
        # sample([batches, n_samples]) 生成 (batches, n_samples) 个独立样本
        # 形状变化: (batches, n_samples) -> (batches, n_samples, 1)
        init_locs = (
            self.init_x_dist.sample([batches, n_samples])
            .to(device=self.device)
            .unsqueeze(2)
        )
         # 创建全0循环状态，形状(batches, n_samples, r_length)
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

        # 组合新状态并断开梯度
        # pt.cat: 拼接新位置 (new_pos) 和模式信息 (r)，形成完整状态
        # .detach(): 创建不跟踪梯度的新张量，阻止反向传播通过采样操作
        #
        # 为什么需要 detach？
        #   标准粒子滤波流程: 采样 → 权重计算 → 重采样 → 下一时刻
        #   问题: 如果保留梯度，重采样（离散操作）会阻断梯度传播，导致无法端到端训练
        #   解决方案: 在提议分布采样后 detach()，断开梯度链
        #
        # 梯度流分析:
        #   情况1: 不加 detach()
        #     x_t_1 → [M_t_proposal] → x_t → [权重计算] → 损失
        #        ↑___________________________↓
        #              梯度可以回传
        #     问题: 经过重采样后梯度链断裂，训练不稳定
        #
        #   情况2: 加 detach()
        #     x_t_1 → [M_t_proposal] → x_t.detach() → [权重计算] → 损失
        #        ↑___________________________________________↓
        #              梯度直接回传到 x_t_1，跳过采样操作
        #     优势: 训练稳定，符合粒子滤波的数学原理
        #
        # 数学意义:
        #   提议分布采样 x_t ~ M_t(·|x_{t-1}) 是随机操作，不可微分
        #   使用 .detach() 明确标记这一点，梯度通过其他路径（如权重计算）回传
        return pt.cat((new_pos, r), dim=2).detach()

    def log_M_t(self, k, x_t, x_t_1, t: int):
        """
        提议分布的对数密度

        计算状态转移的对数概率密度 log M_t(x_t | x_{t-1})。

        核心要点：
        -----------
        1. 用途：重要性采样权重修正。在非 Bootstrap PF 中，提议分布 M_t 与先验 P_t 不同，
           需要计算 log M_t 来修正权重：log w_t = log p(y|x) + log p(x|x') - log M_t(x|x')

        2. 假设：提议分布是高斯分布 N(x_t; μ_t, σ²)，其中 μ_t = a[k] * x_{t-1} + b[k]

        3. 与 M_t_proposal 的关系：
           - M_t_proposal: 采样新状态（随机，不可微，返回 x_t）
           - log_M_t: 计算该样本的密度（确定性，可微，返回 log_prob）

        4. 使用场景：
           - Bootstrap PF（默认）：M_t = P_t，不需要此方法（比值为1）
           - Guided PF：M_t ≠ P_t，必须使用此方法计算权重修正

        数学公式：
        ----------
        log M_t = -1/(2σ²) * (x_t - μ_t)² - 1/2 * log(2πσ²)
                = var_factor * (x_t - locs)² + pre_factor
        """
        # 获取模型 k 的线性动态参数
        scaling = self.a[k]  # 缩放系数 a[k]
        bias = self.b[k]     # 偏置 b[k]

        # 计算预测的均值位置 μ_t = a[k] * x_{t-1} + b[k]
        locs = scaling * x_t_1[:, :, 0] + bias

        # 计算高斯对数密度：var_factor * (x_t - μ_t)² + pre_factor
        # var_factor = -1/(2σ²), pre_factor = -1/2 * log(2πσ²)
        return self.var_factor * ((x_t[:, :, 0] - locs) ** 2) + self.pre_factor

    def log_eta_t(self, x_t, t: int):
        """辅助权重"""
        pass

    def log_R_0(self, x_0):
        """时间0的Radon-Nikodym导数"""
        return pt.zeros([x_0.size(0), x_0.size(1)], device=self.device)

    def log_R_t(self, x_t, x_t_1, t: int):
        """
        时间t的Radon-Nikodym导数（对数权重校正）

        整体功能：
        --------
        计算粒子滤波中重要性采样权重校正的对数形式（log 权重），
        用于在时间步 t 对粒子权重进行修正，保证估计的无偏性。

        数学背景：
        --------
        权重校正公式：
            log w_t = log p(y_t|x_t) + log p(x_t|x_{t-1}) - log M_t(x_t|x_{t-1})

        其中：
            - log p(y_t|x_t)：观测似然
            - log p(x_t|x_{t-1})：先验转移概率（由 switching_dyn.get_weight 提供）
            - log M_t(x_t|x_{t-1})：提议分布密度（由 log_M_t 提供）

        代码逐行解析：
        ------------
        第1步：prop_density = self.log_M_t(x_t, x_t_1, t)
            - log_M_t 是提议分布的对数密度（proposal distribution）
            - 评估从 x_{t-1} 采样得到 x_t 的"合理性"
            - prop_density 是一个携带梯度的张量

        第2步：return ( switching_dyn.get_weight(...) + prop_density - prop_density.detach() )
            - switching_dyn.get_weight(...)：模式切换网络的权重，评估模式转移合理性
            - prop_density：提议分布密度 log M_t（带梯度）
            - prop_density.detach()：提议分布密度（无梯度）

        核心技巧：prop_density - prop_density.detach()
        --------------------------------------------
        前向传播（计算输出）：
            prop_density - prop_density.detach() = 0
            在实际计算中相当于"加 0"，不改变输出值

        反向传播（计算梯度）：
            梯度(prop_density) - 梯度(prop_density.detach())
            = 梯度(prop_density) - 0
            = 梯度(prop_density)
            梯度能正常反向传播回去，优化切换网络的参数

        通俗比喻：
            就像一张"门票"：允许"信息"（梯度）通过，但不允许"人"（数值本身）通过。
            在前向计算时相当于"透明"的，不改变任何东西；
            但在反向时允许梯度"透视"过来。

        为什么需要这个技巧？
        -------------------
        在 DIMMPF 的端到端训练中：
        - switching_dyn 是一个可学习的神经网络
        - log_M_t 提议分布来自真实的状态转移模型
        - 我们希望：
          1. 前向计算：不重复计算提议分布（避免数值误差）
          2. 反向梯度：让梯度能够流过提议分布，优化切换策略

        数学意义：
            1. 权重校正：只包含模式切换权重，不重复计算提议分布
            2. 梯度传播：允许梯度通过提议分布传播，支持端到端训练
            3. 数值稳定性：避免重复计算导致的数值误差

        在粒子滤波流程中的位置：
            时间步 t-1 → [提议分布 M_t] → 时间步 t
                                  ↓
                        计算 log_R_t 进行权重校正
                                  ↓
                        更新粒子权重 → 重采样

        这是一种非常巧妙的实现技巧，确保了DIMMPF算法能够正确进行梯度训练。
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

        # =========================================================================
        # 1. 创建动态网络和观测网络（核心创新）
        # =========================================================================
        # DIMMPF 的核心创新：使用神经网络替代线性方程，实现端到端学习
        #
        # Simple_NN 结构（推测）：
        #   输入层 (1) ──► 隐藏层 (hidden_size) ──► ... ──► 输出层 (1)
        #        ↑              ↑                        ↑
        #      状态x_{t-1}    ReLU激活                 预测位置x_t
        #
        # 为每个模型创建独立的网络，实现真正的多模型并行：
        #   - 模型0：学习匀速运动动态
        #   - 模型1：学习加速运动动态
        #   - 模型2：学习转弯运动动态
        #   - ...
        #
        # 为什么使用 ModuleList 而不是普通列表？
        #   - 普通列表：PyTorch 不知道这些网络的存在，参数不会被优化
        #   - ModuleList：自动注册所有子模块参数，支持 .parameters()、.cuda() 等
        #   - 支持反向传播和梯度优化
        #   - 支持设备管理（.to(device) 时自动移动所有模块）
        #
        # 与 RLPF 的区别：
        #   - RLPF：状态转移 = 线性方程 a·x + b（固定参数）
        #   - DIMMPF：状态转移 = 神经网络 f_θ(x)（可学习）
        #
        # 与 IMMPF 的区别：
        #   - IMMPF：使用预定义模型（CV、CA、CT 等）
        #   - DIMMPF：使用神经网络自动学习模型动态
        self.dyn_models = pt.nn.ModuleList(
            [Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)]
        )
        self.obs_models = pt.nn.ModuleList(
            [Simple_NN(1, hidden_size, 1, layers) for _ in range(n_models)]
        )

        # =========================================================================
        # 2. 切换动力学网络
        # =========================================================================
        # 外部传入的神经网络，负责学习模式切换概率
        # 功能接口：
        #   - R_0(batches, n_samples, k): 初始模式分布
        #   - R_t(x_t_1[:,:,1:], k): 转移模式分布
        #   - get_regime_probs(x_t[:,:,1:]): 获取模式概率
        self.switching_dyn = switching_dyn

        # =========================================================================
        # 3. 参数初始化缩放
        # =========================================================================
        # 对所有可学习参数进行缩放，控制初始输出幅度
        #
        # init_scale 取值影响：
        #   - 1.0 (默认): 标准初始化，适用于大多数情况
        #   - 0.1: 小权重初始化，防止梯度爆炸，训练更稳定
        #   - 10.0: 大权重初始化，需要强初始信号时使用
        #
        # ⚠️ 注意：当前代码存在潜在问题！
        #   当前: p = p * init_scale  ← 创建了新张量，未修改原参数！
        #   正确: p.data *= init_scale 或 p.mul_(init_scale)
        #
        # self.parameters() 返回所有可学习参数（递归遍历所有子模块）：
        #   - dyn_models[0..n-1] 的所有权重和偏置
        #   - obs_models[0..n-1] 的所有权重和偏置
        #   - switching_dyn 的所有参数
        for p in self.parameters():
            p = p * init_scale

        # =========================================================================
        # 4. 可学习的噪声参数（创新点）
        # =========================================================================
        # 传统方法：噪声方差是超参数，需要手动调参
        # DIMMPF：噪声参数也是可学习的，自动从数据中学习最优噪声水平
        #
        # sd_d: 过程噪声标准差（状态转移噪声）
        #   - 初始化: Uniform(0.1, 0.5)
        #   - 影响: 控制状态转移的随机性
        #   - 学习: 根据数据自动调整
        #
        # sd_o: 观测噪声标准差
        #   - 初始化: Uniform(0.1, 0.5)
        #   - 影响: 控制观测的不确定性
        #   - 学习: 根据数据自动调整
        #
        # 使用 nn.Parameter 包装，使其成为可学习参数
        # 可以通过梯度下降自动优化
        self.sd_d = pt.nn.Parameter(pt.rand(1) * 0.4 + 0.1)
        self.sd_o = pt.nn.Parameter(pt.rand(1) * 0.4 + 0.1)

        # =========================================================================
        # 5. 概率分布定义
        # =========================================================================
        # x_dist: 标准正态分布 N(0, 1)
        #   - 用于采样过程噪声
        #   - 实际噪声 = x_dist.sample() * sd_d
        #
        # init_x_dist: 均匀分布 U(-0.5, 0.5)
        #   - 用于采样初始位置
        #   - 表示对初始状态的完全不确定性
        self.x_dist = pt.distributions.Normal(pt.zeros(1), 1)
        self.init_x_dist = pt.distributions.Uniform(-0.5, 0.5)

        # =========================================================================
        # 6. 预计算常数（优化）
        # =========================================================================
        # pi_fact = (1/2) * log(2π)
        #   - 用于高斯对数似然的快速计算
        #   - 避免重复计算这个常数
        #
        # 高斯对数似然公式：
        #   log N(x; μ, σ²) = -1/2*log(2πσ²) - (x-μ)²/(2σ²)
        #                   = -log(σ) - pi_fact - (x-μ)²/(2σ²)
        self.pi_fact = (1 / 2) * pt.log(pt.tensor(2 * pt.pi))

        # =========================================================================
        # 7. 滤波算法选择
        # =========================================================================
        # 使用 Bootstrap 粒子滤波
        #   - 提议分布 = 先验分布（最简单形式）
        #   - 权重 = 观测似然
        #   - 实现简单，但可能效率较低
        #
        # 备选：Guided PF（需要额外实现引导函数）
        #   - 使用观测信息引导提议
        #   - 效率更高，但实现更复杂
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
        # - self.x_loc / / self.x_scale 标准化：减去均值，除以标准差
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
        
        =====================================================================
        功能：计算粒子从 x_{t-1} 变到 x_t 的"合理性"（对数密度）
        
        在粒子滤波中，粒子怎么从上一时刻状态变到当前时刻？
        这个变化需要一个数学公式描述，即"提议分布" q(x_t | x_{t-1})
        
        log_M_t 计算：给定粒子从 x_{t-1} 变到 x_t，这个变化有多"合理"？
        =====================================================================
        
        通俗比喻（开车导航）：
        ---------------------------------------------------------------------
        - locs = GPS 导航告诉你"下一个路口应该在坐标 (5, 3)"
        - x_t = 你实际停在了坐标 (5.1, 3.2)
        - (x_t - locs) ** 2 = 你的实际位置和导航预测的差距平方
        - var_factor_dyn = 惩罚系数（差距越大，惩罚越重）
        - pre_factor_dyn = 基础分数（保证结果是对数概率）
        
        log_M_t 返回的就是："你停在这个位置有多合理？"（用对数分数表示）
        =====================================================================
        
        数学公式（高斯分布的对数密度）：
        ---------------------------------------------------------------------
        log p(x_t | x_{t-1}) = -1/(2σ²) * (x_t - μ)² - 1/2 * log(2πσ²)
        
        代码对应：
          - x_t[:, :, None, 0] - locs  → (x_t - μ)：实际位置与预测差距
          - ** 2                        → (x_t - μ)²：差距的平方
          - self.var_factor_dyn         → -1/(2σ²)：高斯公式常数
          - self.pre_factor_dyn         → -1/2 * log(2πσ²)：归一化常数
        =====================================================================
        """
        # 第1步：用模型 k 的动态神经网络预测粒子"应该去哪里"
        # 输入：上一时刻状态 x_{t-1}（取第一维）
        # 输出：预测的下一时刻位置 locs
        locs = self.dyn_models[k](x_t_1[:, :, 0:1]).squeeze()
        
        # 第2步：调整形状，方便后面计算
        # 比喻：把一张二维纸片展开成三维，便于和其他数据对齐
        locs = locs[:, None, :]
        
        # 第3步：计算高斯分布的对数密度
        # 公式：log p(x_t | x_{t-1}) = -1/(2σ²) * (x_t - μ)² - 1/2 * log(2πσ²)
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
