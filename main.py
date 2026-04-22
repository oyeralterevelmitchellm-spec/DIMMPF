# =============================================================================
# DIMMPF 项目主入口文件
# 本文件是整个项目的入口点，负责解析命令行参数、初始化实验、执行训练和测试
# =============================================================================

# 导入必要的库
import argparse  # 用于解析命令行参数
from Net import (  # 从 Net.py 导入各种模型类
    PF,  # 基础粒子滤波模型 (Particle Filter)
    RLPF,  # 递归线性粒子滤波 (Recursive Linear Particle Filter)
    Redefined_RLPF,  # 重定义的 RLPF，用于计算 ELBO 损失
    Markov_Switching,  # 马尔可夫切换模型
    Polya_Switching,  # 波利亚切换模型
    NN_Switching,  # 神经网络切换模型
    Erlang_Switching,  # 爱尔朗切换模型
    LSTM,  # 长短期记忆网络
    Transformer,  # Transformer 模型
    DIMMPF,  # 可微分交互多模型粒子滤波 (核心算法)
    DIMMPF_redefined,  # 重定义的 DIMMPF，用于计算 ELBO 损失
    IMMPF,  # 交互多模型粒子滤波 (对比算法)
)
from dpf_rs.model import Simulated_Object, State_Space_Dataset  # 数据相关类
from trainingRS import test, e2e_train, train_s2s  # 训练和测试函数
from dpf_rs.simulation import Differentiable_Particle_Filter  # 可微分粒子滤波器
from simulationRS import IMM_Particle_Filter  # IMM 粒子滤波器
from dpf_rs.resampling import Soft_Resampler_Systematic, OT_Resampler  # 重采样器
from dpf_rs.loss import Supervised_L2_Loss, Magnitude_Loss  # 损失函数
from dpf_rs.results import Log_Likelihood_Factors  # 结果记录
import torch as pt  # PyTorch 深度学习框架
from dpf_rs.utils import aggregate_runs, fix_rng  # 工具函数
import pickle  # 用于保存结果
import numpy as np  # 数值计算库
import time  # 时间相关函数


def main():
    """
    主函数：解析命令行参数并初始化实验
    
    整体流程：
    1. 定义命令行参数
    2. 根据参数创建模拟数据
    3. 根据选择的算法初始化模型
    4. 执行训练和测试
    5. 保存结果
    """
    
    # =========================================================================
    # 第一部分：命令行参数定义
    # 使用 argparse 库定义所有可配置的参数
    # =========================================================================
    
    # 创建参数解析器
    # argparse 是 Python 标准库，用于解析命令行参数
    # 让用户可以通过终端输入配置，如：python main.py --device cpu --alg RLPF
    parser = argparse.ArgumentParser(description="Testing")
    
    # 参数1: 设备选择 (CPU 或 GPU)
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="device to use",
    )
    
    # 参数2: 算法选择
    # 可选算法：RLPF, LSTM, Transformer, DIMMPF, DIMMPF-OT, DIMMPF-N, IMMPF
    parser.add_argument(
        "--alg",
        dest="alg",
        type=str,
        default="DIMMPF",
        choices=["RLPF", "LSTM", "Transformer", "DIMMPF", "DIMMPF-OT", "DIMMPF-N", "IMMPF"],
        help="algorithm to use",
    )
    
    # 参数3: 实验类型选择
    # Markov: 马尔可夫切换, Polya: 波利亚切换, Exchange: 交换模型, Erlang: 爱尔朗切换
    parser.add_argument(
        "--experiment",
        dest="experiment",
        type=str,
        default="Markov",
        choices=["Markov", "Polya", "Exchange", "Erlang"],
        help="Experiment to run",
    )
    
    # 参数4: 学习率 (Learning Rate)
    # 控制参数更新的步长，太大可能导致不稳定，太小收敛慢
    parser.add_argument(
        "--lr", dest="lr", type=float, default=0.05, help="Initial max learning rate"
    )
    
    # 参数5: 权重衰减 (Weight Decay)
    # 正则化项，防止过拟合
    parser.add_argument(
        "--w_decay", dest="w_decay", type=float, default=0.05, help="Weight decay strength"
    )
    
    # 参数6: 学习率衰减步数
    # 在哪些 epoch 降低学习率，如 [10, 20, 30, 40] 表示在第 10, 20, 30, 40 个 epoch 降低学习率
    parser.add_argument(
        "--lr_steps",
        dest="lr_steps",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40],
        help="steps to decrease the lr",
    )
    
    # 参数7: 学习率衰减因子
    # 每次衰减时，学习率乘以这个因子，如 0.5 表示每次降低为原来的一半
    parser.add_argument(
        "--lr_gamma", dest="lr_gamma", type=float, default=0.5, help="learning rate decay per step"
    )
    
    # 参数8: 梯度裁剪值
    # 防止梯度爆炸，将梯度限制在这个范围内，如 10 表示限制在 [-10, 10]
    parser.add_argument(
        "--clip", dest="clip", type=float, default=10, help="Value to clip the gradient at"
    )
    
    # 参数9: 损失函数权重 lambda
    # 控制 ELBO 损失和 MSE 损失的比例
    # 训练损失 = MSE + λ × ELBO，λ 越小越侧重预测准确性
    parser.add_argument(
        "--lamb", dest="lamb", type=float, default=0.02, help="Ratio of ELBO to MSE loss"
    )
    
    # 参数10: 结果保存位置
    # 指定实验结果保存的文件名（不含扩展名），保存在 ./results/ 目录下
    parser.add_argument(
        "--store_loc",
        dest="store_loc",
        type=str,
        default="temp",
        help="File in the results folder to store the results dictionary",
    )
    
    # 参数11: 重复运行次数
    # 为了获得稳定的结果，多次运行取平均
    parser.add_argument(
        "--n_runs", dest="n_runs", type=int, default=20, help="Number of runs to average"
    )
    
    # 参数12: 神经网络层数
    # 用于 RLPF 和 DIMMPF 中的神经网络
    parser.add_argument(
        "--layers",
        dest="layers",
        type=int,
        default=3,
        help="Number of fully connected layers in neural networks",
    )
    
    # 参数13: 隐藏层大小
    # 神经网络隐藏层的节点数
    parser.add_argument(
        "--hid_size",
        dest="hidden_size",
        type=int,
        default=11,
        help="Number of nodes in hidden layers",
    )
    
    # 参数14: 数据目录
    # 指定模拟数据保存的目录名，保存在 ./data/{data_dir}/ 目录下
    parser.add_argument(
        "--data_dir", dest="data_dir", type=str, default="temp", help="Data directory"
    )
    
    # 参数15: 训练轮数
    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=50, help="Number of epochs to train for"
    )

    # 解析命令行参数
    # 读取用户在命令行输入的所有参数，返回包含所有参数值的对象
    args = parser.parse_args()

    # =========================================================================
    # 第二部分：数据创建函数
    # 根据实验类型创建模拟数据
    # =========================================================================
    
    def create_data():
        """
        创建模拟数据的函数
        
        根据 args.experiment 的值选择不同的切换动力学模型
        然后使用 PF 模型生成模拟数据
        
        参数说明：
        - n_models=8: 8个模型
        - a=[-0.1, -0.3, ...]: 状态转移的缩放系数
        - b=[0, -2, ...]: 状态转移的偏置
        - var_s=0.1: 噪声方差
        """
        # nonlocal 关键字：在嵌套函数中使用外部函数的变量
        # 让内部函数可以读取和修改外部函数的 args 变量
        nonlocal args
        
        # 根据实验类型选择切换模型
        if args.experiment == "Markov":
            # 马尔可夫切换：使用转移概率矩阵
            # 8: 模型数量, 0.8: 保持当前模型的概率, 0.15: 切换到相邻模型的概率
            switching = Markov_Switching(8, 0.8, 0.15, "Boot", device=args.device)
        elif args.experiment == "Polya":
            # 波利亚切换：基于计数器的切换
            switching = Polya_Switching(8, "Boot", args.device)
        else:
            # 爱尔朗切换：基于爱尔朗分布的切换
            switching = Erlang_Switching(8, "Boot", args.device)
        
        # 创建 PF 模型用于生成数据
        # 状态转移方程：x_t = a[k] * x_{t-1} + b[k] + noise
        model = PF(
            [-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9],  # 8个模型的 a 值
            [0, -2, 2, -4, 0, 2, -2, 4],  # 8个模型的 b 值
            0.1,  # 噪声方差
            switching,  # 切换模型
            "Boot",  # Bootstrap 滤波
            args.device,
        )
        
        # 创建模拟对象
        # 参数：model, 序列长度=100, 批量大小=100, 状态维度=1, 设备
        sim_obj = Simulated_Object(model, 100, 100, 1, args.device)
        
        # 保存数据
        # 参数：保存路径, 训练序列数=50, 测试序列数=20, 文件前缀, 自动覆盖
        # 生成的文件：train_states.pt, train_obs.pt, test_states.pt, test_obs.pt
        sim_obj.save(f"./data/{args.data_dir}", 50, 20, "", bypass_ask=True)

    # =========================================================================
    # 第三部分：算法初始化与训练
    # 根据用户选择的算法，定义对应的训练和测试函数
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 3.1 IMMPF 算法 (第 163-204 行)
    # 交互多模型粒子滤波 (Interacting Multiple Model Particle Filter)
    # 这是传统的 IMM-PF 算法，用于对比
    # -------------------------------------------------------------------------
    if args.alg == "IMMPF":

        def train_test():
            """
            IMMPF 的训练和测试函数
            
            特点：
            - 不需要训练，直接测试
            - 使用 2000 个粒子
            - 使用 Soft Resampler
            """
            nonlocal args
            
            # 加载数据集
            # lazy=False: 一次性加载所有数据到内存
            # num_workers=0: 不使用多进程加载数据
            data = State_Space_Dataset(
                f"./data/{args.data_dir}", lazy=False, device=args.device, num_workers=0
            )
            
            # 根据实验类型创建对应的 IMMPF 模型
            if args.experiment == "Markov":
                model = IMMPF(
                    [-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9],
                    [0, -2, 2, -4, 0, 2, -2, 4],
                    0.1,
                    Markov_Switching(8, 0.8, 0.15, "Boot", device=args.device),
                    args.device,
                )
            elif args.experiment == "Polya":
                model = IMMPF(
                    [-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9],
                    [0, -2, 2, -4, 0, 2, -2, 4],
                    0.1,
                    Polya_Switching(8, "Boot", device=args.device),
                    args.device,
                )
            else:
                model = IMMPF(
                    [-0.1, -0.3, -0.5, -0.9, 0.1, 0.3, 0.5, 0.9],
                    [0, -2, 2, -4, 0, 2, -2, 4],
                    0.1,
                    Erlang_Switching(8, "Boot", device=args.device),
                    args.device,
                )
            
            # 创建 IMM 粒子滤波器（传统算法，不需要训练）
            # 参数说明：
            #   model: IMMPF 模型（包含 8 个动态模型和切换逻辑）
            #   2000: 粒子数量（使用 2000 个粒子，比学习算法多，因为不需要训练）
            #   Soft_Resampler_Systematic(1, 0, args.device): 软系统重采样器
            #       - 1: 温度参数 α
            #       - 0: 偏置参数 β
            #       - args.device: 运行设备
            #   2001: ESS 阈值（有效样本量阈值）
            #       - 大于粒子数 2000，表示每步都进行重采样
            #   args.device: 运行设备（cuda/cpu）
            #   "normal": IMM 类型，使用标准的 IMM 实现（非论文新方法）
            # 特点：
            #   - 不需要训练，直接测试
            #   - 使用预定义的切换模型（Markov/Polya/Erlang）
            #   - 作为对比基准，验证 DIMMPF 的改进效果
            DPF = IMM_Particle_Filter(
                model,
                2000,
                Soft_Resampler_Systematic(1, 0, args.device),
                2001,
                args.device,
                "normal",
            )
            
            # 定义损失函数：监督 L2 损失（均方误差）
            # 参数说明：
            #   function=lambda x: x[:, :, 0:1]: 提取状态的第一个维度
            #       - x[:, :, 0:1] 表示取所有批次、所有时间步、第 0 个维度
            #       - 因为状态可能是多维的，这里只取第一个维度进行评估
            # 作用：评估预测状态与真实状态之间的误差
            loss = Supervised_L2_Loss(function=lambda x: x[:, :, 0:1])
            
            # 执行测试
            # 参数说明：
            #   DPF: IMM 粒子滤波器
            #   loss: L2 损失函数
            #   50: 时间步数（测试 50 个时间步）
            #   data: 数据集（State_Space_Dataset）
            #   -1: 测试数据量（-1 表示使用所有测试数据）
            #   0.25: 测试比例（使用 25% 的数据作为测试集）
            # 返回：测试结果字典，包含损失、时间等指标
            return test(DPF, loss, 50, data, -1, 0.25)

    # -------------------------------------------------------------------------
    # 3.2 RLPF 算法 (第 205-257 行)
    # 递归线性粒子滤波 (Recursive Linear Particle Filter)
    # 使用神经网络学习切换动力学
    # -------------------------------------------------------------------------
    if args.alg == "RLPF":

        def train_test():
            """
            RLPF 的训练和测试函数
            
            特点：
            - 使用神经网络学习切换动力学
            - 需要端到端训练
            - 使用 200 个粒子
            """
            nonlocal args
            
            # 加载数据集
            data = State_Space_Dataset(
                f"./data/{args.data_dir}", lazy=False, device=args.device, num_workers=0
            )
            
            # 创建 RLPF 模型（递归线性粒子滤波）
            # RLPF 特点：
            #   - 使用神经网络学习切换动力学（替代预定义的切换模型）
            #   - 保持状态转移方程的线性结构
            #   - 端到端训练所有参数
            # 参数说明：
            #   8: 模型数量（8个候选动态模型）
            #   NN_Switching(8, 8, "Uni", args.device, 0): 切换模型（神经网络学习）
            #       - 8: 输入维度（模型数）
            #       - 8: 输出维度（模型数）
            #       - "Uni": 均匀初始化
            #       - args.device: 运行设备
            #       - 0: 额外参数
            #   1: 初始缩放因子（状态转移的初始缩放）
            #   args.layers: 神经网络层数（默认3层）
            #   args.hidden_size: 隐藏层大小（默认11个节点）
            #   "Uni": 初始化方式（均匀初始化）
            #   args.device: 运行设备（cuda/cpu）
            # 与 IMMPF 的区别：IMMPF 使用预定义切换模型，RLPF 使用神经网络学习
            # 与 DIMMPF 的关系：RLPF 是 DIMMPF 的简化版，只学习切换，不学习状态转移
            model = RLPF(
                8,
                NN_Switching(8, 8, "Uni", args.device, 0),
                1,
                args.layers,
                args.hidden_size,
                "Uni",
                args.device,
            )

            # 创建重定义的模型，用于计算 ELBO 损失
            # 作用：创建一个修改版的 RLPF 模型，专门用于计算 ELBO 损失
            # 原因：ELBO 计算需要不同的前向传播逻辑
            # 特点：共享参数，但输出不同的统计量
            re_model = Redefined_RLPF(model)

            # 创建可微分粒子滤波器（用于端到端训练）
            # 参数说明：
            #   model: RLPF 模型
            #   200: 粒子数量（使用 200 个粒子，比 IMMPF 少，因为需要训练）
            #   Soft_Resampler_Systematic(1, 0.6, args.device): 软系统重采样器
            #       - 1: 温度参数 α
            #       - 0.6: 偏置参数 β
            #       - args.device: 运行设备
            #   100: ESS 阈值（有效样本量阈值）
            #       - 小于粒子数 200，表示只在 ESS 低于 100 时进行重采样
            #   args.device: 运行设备（cuda/cpu）
            # 特点：
            #   - 端到端可微分，可以反向传播梯度
            #   - 使用软重采样，保持梯度流
            DPF = Differentiable_Particle_Filter(
                model,
                200,
                Soft_Resampler_Systematic(1, 0.6, args.device),
                100,
                args.device,
            )

            # 创建重定义的粒子滤波器（用于计算 ELBO）
            # 作用：专门用于计算 ELBO 损失，共享 RLPF 的参数
            # 原因：ELBO 计算需要不同的前向传播逻辑，但参数相同
            # 特点：
            #   - 共享 RLPF 的参数（通过 re_model 引用）
            #   - 输出对数似然因子，用于计算 ELBO
            #   - 不用于训练，只用于计算损失
            DPF_re = Differentiable_Particle_Filter(
                re_model,
                200,
                Soft_Resampler_Systematic(1, 0, args.device),
                100,
                args.device,
            )

            # 定义优化器（Adam）
            # 参数说明：
            #   DPF.parameters(): 需要优化的参数（RLPF 的所有可学习参数）
            #   lr=args.lr: 初始学习率（默认 0.05）
            #   weight_decay=args.w_decay: 权重衰减系数（默认 0.05）
            # 作用：使用 Adam 优化算法更新模型参数
            opt = pt.optim.Adam(DPF.parameters(), lr=args.lr, weight_decay=args.w_decay)

            # 定义学习率调度器（MultiStepLR）
            # 参数说明：
            #   opt: 优化器对象
            #   milestones=args.lr_steps: 学习率衰减的 epoch 列表（默认 [10, 20, 30, 40]）
            #   gamma=args.lr_gamma: 衰减因子（默认 0.5）
            # 作用：在指定的 epoch 降低学习率，帮助模型更好地收敛
            opt_schedule = pt.optim.lr_scheduler.MultiStepLR(
                opt, milestones=args.lr_steps, gamma=args.lr_gamma
            )

            # 定义损失函数：监督 L2 损失（均方误差）
            # 参数说明：
            #   function=lambda x: x[:, :, 0:1]: 提取状态的第一个维度
            # 作用：评估预测状态与真实状态之间的误差
            loss = Supervised_L2_Loss(function=lambda x: x[:, :, 0:1])

            # 执行端到端训练
            # 参数说明：
            #   DPF: 主粒子滤波器（用于训练）
            #   DPF_re: 重定义的粒子滤波器（用于计算 ELBO）
            #   opt: 优化器
            #   loss: 损失函数
            #   50: 时间步数
            #   data: 数据集
            #   None: 测试数据集（None 表示从训练数据中划分）
            #   [10, 10, 10]: 批量大小 [训练, 验证, 测试]
            #   [0.7, 0.15, 0.15]: 数据划分比例 [训练, 验证, 测试]
            #   args.epochs: 训练轮数（默认 50）
            #   10: 测试时粒子数放大倍数
            #   opt_schedule: 学习率调度器
            #   True: 是否打印详细信息
            #   args.clip: 梯度裁剪值（默认 10）
            #   True: 是否进行数据标准化
            #   args.lamb: ELBO 损失权重（默认 0.02）
            # 返回：训练结果字典，包含损失、时间等指标
            return e2e_train(
                DPF,
                DPF_re,
                opt,
                loss,
                50,
                data,
                None,
                [10, 10, 10],
                [0.7, 0.15, 0.15],
                args.epochs,
                10,
                opt_schedule,
                True,
                args.clip,
                True,
                args.lamb,
            )

    # -------------------------------------------------------------------------
    # 3.3 LSTM 算法 (第 258-303 行)
    # 长短期记忆网络 (Long Short-Term Memory)
    # 使用 LSTM 直接映射观测到状态
    # -------------------------------------------------------------------------
    if args.alg == "LSTM":

        def train_test():
            """
            LSTM 的训练和测试函数
            
            特点：
            - 使用 LSTM 网络直接学习观测到状态的映射
            - 不需要粒子滤波
            - 作为基线方法
            """
            nonlocal args
            
            # 加载数据集
            data = State_Space_Dataset(
                f"./data/{args.data_dir}", lazy=False, device=args.device, num_workers=0
            )

            # 创建 LSTM 模型
            # 参数说明：
            #   1: 观测维度（输入维度）
            #   10: 隐藏层维度
            #   1: 状态维度（输出维度）
            #   2: LSTM 层数
            #   args.device: 运行设备
            # 作用：创建 LSTM 网络，将观测序列映射到状态序列
            LSTM_model = LSTM(1, 10, 1, 2, args.device)

            # 定义优化器（Adam）
            # 参数说明：
            #   LSTM_model.parameters(): LSTM 的所有可学习参数
            #   lr=args.lr: 初始学习率（默认 0.05）
            #   weight_decay=args.w_decay: 权重衰减系数（默认 0.05）
            opt = pt.optim.Adam(
                LSTM_model.parameters(), lr=args.lr, weight_decay=args.w_decay
            )

            # 定义学习率调度器
            # 参数说明：
            #   opt: 优化器对象
            #   milestones=args.lr_steps: 学习率衰减的 epoch 列表
            #   gamma=args.lr_gamma: 衰减因子（默认 0.5）
            opt_schedule = pt.optim.lr_scheduler.MultiStepLR(
                opt, milestones=args.lr_steps, gamma=args.lr_gamma
            )

            # 执行序列到序列训练
            # 参数说明：
            #   LSTM_model: LSTM 模型
            #   opt: 优化器
            #   data: 数据集
            #   [10, 10, 10]: 批量大小 [训练, 验证, 测试]
            #   [0.7, 0.15, 0.15]: 数据划分比例 [训练, 验证, 测试]
            #   args.epochs: 训练轮数（默认 50）
            #   opt_schedule: 学习率调度器
            #   True: 是否打印详细信息
            #   args.clip: 梯度裁剪值（默认 10）
            # 返回：训练结果字典
            return train_s2s(
                LSTM_model,
                opt,
                data,
                [10, 10, 10],
                [0.7, 0.15, 0.15],
                args.epochs,
                opt_schedule,
                True,
                args.clip,
            )

    # -------------------------------------------------------------------------
    # 3.4 Transformer 算法 (第 304-349 行)
    # Transformer 模型
    # 使用 Transformer 直接映射观测到状态
    # -------------------------------------------------------------------------
    if args.alg == "Transformer":

        def train_test():
            """
            Transformer 的训练和测试函数
            
            特点：
            - 使用 Transformer 网络直接学习观测到状态的映射
            - 不需要粒子滤波
            - 作为基线方法
            """
            nonlocal args
            
            # 加载数据集
            data = State_Space_Dataset(
                f"./data/{args.data_dir}", lazy=False, device=args.device, num_workers=0
            )

            # 创建 Transformer 模型
            # 参数说明：
            #   1: 观测维度（输入维度）
            #   10: 隐藏层维度
            #   1: 状态维度（输出维度）
            #   50: 序列长度
            #   args.device: 运行设备
            #   2: Transformer 层数
            # 作用：创建 Transformer 网络，将观测序列映射到状态序列
            Transformer_model = Transformer(1, 10, 1, 50, args.device, 2)

            # 定义优化器（Adam）
            # 参数说明：
            #   Transformer_model.parameters(): Transformer 的所有可学习参数
            #   lr=args.lr: 初始学习率（默认 0.05）
            #   weight_decay=args.w_decay: 权重衰减系数（默认 0.05）
            opt = pt.optim.Adam(
                Transformer_model.parameters(), lr=args.lr, weight_decay=args.w_decay
            )

            # 定义学习率调度器
            # 参数说明：
            #   opt: 优化器对象
            #   milestones=args.lr_steps: 学习率衰减的 epoch 列表
            #   gamma=args.lr_gamma: 衰减因子（默认 0.5）
            opt_schedule = pt.optim.lr_scheduler.MultiStepLR(
                opt, milestones=args.lr_steps, gamma=args.lr_gamma
            )

            # 执行序列到序列训练
            # 参数说明：
            #   Transformer_model: Transformer 模型
            #   opt: 优化器
            #   data: 数据集
            #   [10, 10, 10]: 批量大小 [训练, 验证, 测试]
            #   [0.7, 0.15, 0.15]: 数据划分比例 [训练, 验证, 测试]
            #   args.epochs: 训练轮数（默认 50）
            #   opt_schedule: 学习率调度器
            #   True: 是否打印详细信息
            #   args.clip: 梯度裁剪值（默认 10）
            # 返回：训练结果字典
            return train_s2s(
                Transformer_model,
                opt,
                data,
                [10, 10, 10],
                [0.7, 0.15, 0.15],
                args.epochs,
                opt_schedule,
                True,
                args.clip,
            )

    # -------------------------------------------------------------------------
    # 3.5 DIMMPF 算法 (第 350-402 行)
    # 可微分交互多模型粒子滤波 (Deep Interacting Multiple Model Particle Filter)
    # 论文的核心算法，使用神经网络学习切换动力学和状态转移
    # -------------------------------------------------------------------------
    if args.alg == "DIMMPF":

        def train_test():
            """
            DIMMPF 的训练和测试函数
            
            特点：
            - 使用神经网络学习切换动力学
            - 使用神经网络学习状态转移（每个模型一个网络）
            - 端到端训练所有参数
            - 使用 200 个粒子
            """
            nonlocal args
            
            # 加载数据集
            data = State_Space_Dataset(
                f"./data/{args.data_dir}", lazy=False, device=args.device, num_workers=0
            )

            # 创建 DIMMPF 模型（可微分交互多模型粒子滤波）
            # DIMMPF 特点：
            #   - 使用神经网络学习切换动力学（替代预定义的切换模型）
            #   - 使用神经网络学习状态转移（每个模型一个网络）
            #   - 端到端训练所有参数
            # 参数说明：
            #   8: 模型数量（8个候选动态模型）
            #   NN_Switching(8, 8, "Uni", args.device, 0.1): 切换模型（神经网络学习）
            #       - 8: 输入维度（模型数）
            #       - 8: 循环层维度
            #       - "Uni": 均匀初始化
            #       - args.device: 运行设备
            #       - 0.1: 软化参数
            #   1: 初始缩放因子
            #   args.layers: 神经网络层数（默认3层）
            #   args.hidden_size: 隐藏层大小（默认11个节点）
            #   "Uni": 初始化方式
            #   args.device: 运行设备（cuda/cpu）
            # 与 RLPF 的区别：RLPF 只学习切换，DIMMPF 同时学习切换和状态转移
            # 与 IMMPF 的区别：IMMPF 使用预定义模型，DIMMPF 使用神经网络学习
            model = DIMMPF(
                8,
                NN_Switching(8, 8, "Uni", args.device, 0.1),
                1,
                args.layers,
                args.hidden_size,
                "Uni",
                args.device,
            )

            # 创建重定义的模型，用于计算 ELBO 损失
            # 作用：创建一个修改版的 DIMMPF 模型，专门用于计算 ELBO 损失
            # 原因：ELBO 计算需要不同的前向传播逻辑
            # 特点：共享参数，但输出不同的统计量
            re_model = DIMMPF_redefined(model)

            # 创建可微分粒子滤波器（用于端到端训练）
            # 参数说明：
            #   model: DIMMPF 模型
            #   200: 粒子数量
            #   Soft_Resampler_Systematic(1, 0, args.device): 软系统重采样器
            #   100: ESS 阈值
            #   args.device: 运行设备
            #   IMMtype="new": 使用论文提出的新 IMM 实现
            DPF = Differentiable_Particle_Filter(
                model,
                200,
                Soft_Resampler_Systematic(1, 0, args.device),
                100,
                args.device,
            )

            # 创建重定义的粒子滤波器（用于计算 ELBO）
            # 作用：专门用于计算 ELBO 损失
            DPF_re = Differentiable_Particle_Filter(
                re_model,
                200,
                Soft_Resampler_Systematic(1, 0, args.device),
                100,
                args.device,
            )

            # 定义优化器（Adam）
            opt = pt.optim.Adam(DPF.parameters(), lr=args.lr, weight_decay=args.w_decay)

            # 定义学习率调度器
            opt_schedule = pt.optim.lr_scheduler.MultiStepLR(
                opt, milestones=args.lr_steps, gamma=args.lr_gamma
            )

            # 定义损失函数
            loss = Supervised_L2_Loss(function=lambda x: x[:, :, 0:1])

            # 执行端到端训练
            return e2e_train(
                DPF,
                DPF_re,
                opt,
                loss,
                50,
                data,
                None,
                [10, 10, 10],
                [0.7, 0.15, 0.15],
                args.epochs,
                10,
                opt_schedule,
                True,
                args.clip,
                True,
                args.lamb,
            )

    # -------------------------------------------------------------------------
    # 3.6 DIMMPF-OT 算法 (第 403-455 行)
    # 使用最优传输重采样的 DIMMPF
    # -------------------------------------------------------------------------
    if args.alg == "DIMMPF-OT":

        def train_test():
            """
            DIMMPF-OT 的训练和测试函数
            
            特点：
            - 与 DIMMPF 相同，但使用最优传输重采样
            - 最优传输重采样可以更好地保持粒子多样性
            """
            nonlocal args
            
            # 加载数据集
            data = State_Space_Dataset(
                f"./data/{args.data_dir}", lazy=False, device=args.device, num_workers=0
            )

            # 创建 DIMMPF 模型
            model = DIMMPF(
                8,
                NN_Switching(8, 8, "Uni", args.device, 0.1),
                1,
                args.layers,
                args.hidden_size,
                "Uni",
                args.device,
            )

            # 创建重定义的模型
            re_model = DIMMPF_redefined(model)

            # 创建可微分粒子滤波器（使用最优传输重采样）
            # 与 DIMMPF 的区别：使用 OT_Resampler 替代 Soft_Resampler_Systematic
            # OT_Resampler: 最优传输重采样器
            #   - 使用最优传输理论进行重采样
            #   - 可以更好地保持粒子多样性
            #   - 计算成本更高，但性能更好
            DPF = Differentiable_Particle_Filter(
                model,
                200,
                OT_Resampler(args.device),
                100,
                args.device,
                IMMtype="new",
            )

            # 创建重定义的粒子滤波器
            DPF_re = Differentiable_Particle_Filter(
                re_model,
                200,
                OT_Resampler(args.device),
                100,
                args.device,
                IMMtype="new",
            )

            # 定义优化器
            opt = pt.optim.Adam(DPF.parameters(), lr=args.lr, weight_decay=args.w_decay)

            # 定义学习率调度器
            opt_schedule = pt.optim.lr_scheduler.MultiStepLR(
                opt, milestones=args.lr_steps, gamma=args.lr_gamma
            )

            # 定义损失函数
            loss = Supervised_L2_Loss(function=lambda x: x[:, :, 0:1])

            # 执行端到端训练
            return e2e_train(
                DPF,
                DPF_re,
                opt,
                loss,
                50,
                data,
                None,
                [10, 10, 10],
                [0.7, 0.15, 0.15],
                args.epochs,
                10,
                opt_schedule,
                True,
                args.clip,
                True,
                args.lamb,
            )

    # -------------------------------------------------------------------------
    # 3.7 DIMMPF-N 算法 (第 456-508 行)
    # 使用标准 IMM 实现的 DIMMPF
    # -------------------------------------------------------------------------
    if args.alg == "DIMMPF-N":

        def train_test():
            """
            DIMMPF-N 的训练和测试函数
            
            特点：
            - 与 DIMMPF 相同，但使用标准 IMM 实现
            - 用于对比论文提出的新 IMM 实现
            """
            nonlocal args
            
            # 加载数据集
            data = State_Space_Dataset(
                f"./data/{args.data_dir}", lazy=False, device=args.device, num_workers=0
            )

            # 创建 DIMMPF 模型
            model = DIMMPF(
                8,
                NN_Switching(8, 8, "Uni", args.device, 0.1),
                1,
                args.layers,
                args.hidden_size,
                "Uni",
                args.device,
            )

            # 创建重定义的模型
            re_model = DIMMPF_redefined(model)

            # 创建可微分粒子滤波器（使用标准 IMM 实现）
            # 与 DIMMPF 的区别：IMMtype="normal" 使用标准 IMM 实现
            # 标准 IMM 实现：
            #   - 使用传统的 IMM 算法
            #   - 与论文提出的新实现进行对比
            DPF = Differentiable_Particle_Filter(
                model,
                200,
                Soft_Resampler_Systematic(1, 0, args.device),
                100,
                args.device,
                IMMtype="normal",
            )

            # 创建重定义的粒子滤波器
            DPF_re = Differentiable_Particle_Filter(
                re_model,
                200,
                Soft_Resampler_Systematic(1, 0, args.device),
                100,
                args.device,
                IMMtype="normal",
            )

            # 定义优化器
            opt = pt.optim.Adam(DPF.parameters(), lr=args.lr, weight_decay=args.w_decay)

            # 定义学习率调度器
            opt_schedule = pt.optim.lr_scheduler.MultiStepLR(
                opt, milestones=args.lr_steps, gamma=args.lr_gamma
            )

            # 定义损失函数
            loss = Supervised_L2_Loss(function=lambda x: x[:, :, 0:1])

            # 执行端到端训练
            return e2e_train(
                DPF,
                DPF_re,
                opt,
                loss,
                50,
                data,
                None,
                [10, 10, 10],
                [0.7, 0.15, 0.15],
                args.epochs,
                10,
                opt_schedule,
                True,
                args.clip,
                True,
                args.lamb,
            )

    # =========================================================================
    # 第四部分：执行实验 (第 509-897 行)
    # 创建数据、运行实验、保存结果
    # =========================================================================
    
    # 创建模拟数据
    # 调用 create_data() 函数，根据 args.experiment 生成模拟数据
    # 生成的数据保存在 ./data/{args.data_dir}/ 目录下
    create_data()

    # 初始化结果存储列表
    # 用于存储多次运行的结果，最后取平均
    results = []

    # 多次运行实验，取平均结果
    # 原因：神经网络训练有随机性，多次运行可以获得更稳定的结果
    # args.n_runs: 运行次数（默认 20 次）
    for i in range(args.n_runs):
        print(f"Run {i+1}/{args.n_runs}")
        
        # 设置随机种子，确保可复现性
        # fix_rng(i): 设置 PyTorch 和 NumPy 的随机种子为 i
        # 作用：每次运行使用不同的随机种子，但结果可复现
        fix_rng(i)
        
        # 执行训练和测试
        # train_test(): 根据 args.alg 执行对应的训练和测试函数
        # 返回：包含损失、时间等指标的字典
        results.append(train_test())

    # 聚合多次运行的结果
    # aggregate_runs(results): 对多次运行的结果取平均
    # 返回：平均后的结果字典
    results = aggregate_runs(results)

    # 保存结果到文件
    # 保存路径：./results/{args.store_loc}.pickle
    # 格式：Python pickle 格式（序列化对象）
    # 内容：包含损失、每步损失、训练时间、测试时间等指标
    with open(f"./results/{args.store_loc}.pickle", "wb") as f:
        pickle.dump(results, f)

    # 打印最终结果
    # 输出测试损失、训练时间、测试时间等关键指标
    print(f"Final results:")
    print(f"  Test loss: {results['loss']}")
    print(f"  Train time: {results['train_time']}")
    print(f"  Test time: {results['test_time']}")


# 程序入口点
# 当直接运行 main.py 时，执行 main() 函数
# 当作为模块导入时，不执行 main() 函数
if __name__ == "__main__":
    main()
