# =============================================================================
# trainingRS.py - 训练和测试函数
# 本文件包含粒子滤波算法的训练、测试和评估函数
# 
# 文件结构：
# 第一部分：测试函数 - _test() 和 test()
# 第二部分：端到端训练 - e2e_train()（用于RLPF和DIMMPF）
# 第三部分：序列到序列训练 - train_s2s()（用于LSTM和Transformer）
# =============================================================================

import torch as pt
from dpf_rs.simulation import Differentiable_Particle_Filter
from typing import Iterable
import numpy as np
from copy import deepcopy
from dpf_rs.loss import Loss, Compound_Loss, Magnitude_Loss
from dpf_rs import results
import time


# =============================================================================
# 第一部分：测试函数
# 用于评估模型性能
# =============================================================================

def _test(
    DPF: Differentiable_Particle_Filter,
    loss: Loss,
    T: int,
    data: pt.utils.data.DataLoader,
    scale=None,
):
    """
    内部测试函数
    
    执行模型测试，计算测试损失
    
    参数说明：
    ----------
    DPF: Differentiable_Particle_Filter
        粒子滤波器模型
    loss: Loss
        损失函数
    T: int
        时间步数
    data: DataLoader
        测试数据加载器
    scale: tuple, optional
        数据标准化参数 (mean_state, sd_state, mean_obs, sd_obs)
    
    返回：
    ----------
    tuple: (平均损失, 每步损失)
    
    说明：
    - 使用 inference_mode() 禁用梯度计算，提高测试效率
    - 支持数据标准化和反标准化
    - 对于新的IMM实现，临时切换到标准模式进行测试
    """
    # 设置评估模式
    # eval() 会关闭 dropout 和 batch normalization 等训练特定层
    DPF.eval()
    
    # 对于新的IMM实现，临时切换到标准模式进行测试
    # 原因：新的IMM实现可能在训练时使用特殊技巧，测试时需要标准模式
    try:
        IMMtype = DPF.IMMtype
        if IMMtype == "new":
            DPF.IMMtype = "normal"
    except:
        pass
    
    # 禁用梯度计算以提高效率
    # inference_mode() 比 no_grad() 更高效，专为推理设计
    with pt.inference_mode():
        for i, simulated_object in enumerate(data):
            # 数据标准化（如果提供了标准化参数）
            # 标准化公式：x' = (x - mean) / sd
            if not scale is None:
                simulated_object.state -= scale[0]
                simulated_object.observations -= scale[2]
                simulated_object.state /= scale[1]
                simulated_object.observations /= scale[3]
            
            # 清除之前的损失数据
            # 确保每次测试都是独立的，不受之前数据影响
            loss.clear_data()
            
            # 注册真实状态用于计算损失
            # 将真实状态传递给损失函数，用于后续比较
            loss.register_data(truth=simulated_object)
            
            # 运行粒子滤波器
            # 参数：模拟对象，时间步数，损失报告器列表
            DPF(simulated_object, T, loss.get_reporters())
            
            # 计算每步损失
            # per_step_loss() 返回每个时间步的损失值
            loss_t = loss.per_step_loss()
            
            # 反标准化损失
            # 如果数据被标准化过，损失需要乘以标准差的平方
            # 原因：MSE损失与数据尺度的平方成正比
            if not scale is None:
                loss_t = loss_t * (scale[1][0] ** 2)
            
            # 转换为numpy数组
            # detach() 分离计算图，to("cpu") 移到CPU，numpy() 转为numpy数组
            loss_t = loss_t.to(device="cpu").detach().numpy()
    
    # 打印测试损失
    print(f"Test loss: {np.mean(loss_t)}")
    
    # 恢复原来的IMM类型
    try:
        DPF.IMMtype = IMMtype
    except:
        pass
    
    # 返回平均损失和每步损失
    # np.mean(loss_t): 所有时间步的平均损失
    # np.mean(loss_t, axis=0): 每个时间步的平均损失（跨所有批次）
    return np.array([np.mean(loss_t)]), np.mean(loss_t, axis=0)


def test(
    DPF: Differentiable_Particle_Filter,
    loss: Loss,
    T: int,
    data: pt.utils.data.Dataset,
    batch_size: int,
    fraction: float,
):
    """
    测试函数
    
    对模型进行测试，返回测试结果
    
    参数说明：
    ----------
    DPF: Differentiable_Particle_Filter
        粒子滤波器模型
    loss: Loss
        损失函数
    T: int
        时间步数
    data: Dataset
        测试数据集
    batch_size: int
        批量大小，-1表示使用全部数据
    fraction: float
        用于测试的数据比例
    
    返回：
    ----------
    tuple: (平均损失, 每步损失, 训练时间占位符, 测试时间)
    
    说明：
    - 支持按比例划分测试集
    - 使用 DataLoader 批量加载数据
    - 记录测试时间用于性能分析
    """
    # 划分测试集
    # 如果 fraction == 1，使用全部数据
    # 否则，按比例划分（fraction 用于测试，1-fraction 丢弃）
    if fraction == 1:
        test_set = data
    else:
        test_set, _ = pt.utils.data.random_split(data, [fraction, 1 - fraction])
    
    # 设置批量大小
    # -1 表示使用全部数据（单个批次）
    if batch_size == -1:
        batch_size = len(test_set)
    
    # 创建数据加载器
    # 参数说明：
    #   test_set: 测试数据集
    #   min(batch_size, len(test_set)): 实际批量大小（不超过数据集大小）
    #   shuffle=False: 不打乱数据（测试时不需要）
    #   collate_fn=data.collate: 使用数据集的自定义合并函数
    #   num_workers=data.workers: 使用数据集指定的工作进程数
    #   drop_last=True: 丢弃最后一个不完整的批次
    test = pt.utils.data.DataLoader(
        test_set,
        min(batch_size, len(test_set)),
        shuffle=False,
        collate_fn=data.collate,
        num_workers=data.workers,
        drop_last=True,
    )
    
    # 记录测试时间
    start = time.time()
    results = _test(DPF, loss, T, test)
    
    # 返回测试结果和测试时间
    # np.array([0]): 训练时间占位符（测试时没有训练）
    # time.time() - start: 测试耗时
    return *results, np.array([0]), np.array([time.time() - start])


# =============================================================================
# 第二部分：端到端训练函数
# 用于 RLPF 和 DIMMPF 的训练
# 特点：同时优化状态估计和似然估计（ELBO）
# =============================================================================

def e2e_train(
    DPF: Differentiable_Particle_Filter,
    DPF_redefined: Differentiable_Particle_Filter,
    opt: pt.optim.Optimizer,
    loss: Loss,
    T: int,
    data_train: pt.utils.data.Dataset,
    data_test: pt.utils.data.Dataset,
    batch_size: Iterable[int],
    set_fractions: Iterable[float],
    epochs: int,
    test_scaling: float = 1,
    opt_schedule: pt.optim.lr_scheduler.LRScheduler = None,
    verbose: bool = True,
    clip: float = pt.inf,
    normalise=True,
    lam=0.02,
):
    """
    端到端训练函数（用于RLPF和DIMMPF）
    
    执行端到端训练，包括训练、验证和测试
    
    训练流程：
    1. 划分数据集（训练/验证/测试）
    2. 计算数据标准化参数
    3. 创建复合损失函数（MSE + ELBO）
    4. 训练循环：
       - 前向传播（粒子滤波）
       - 计算损失（状态预测误差 + 似然估计）
       - 反向传播
       - 梯度裁剪
       - 参数更新
    5. 验证和保存最佳模型
    6. 测试
    
    参数说明：
    ----------
    DPF: Differentiable_Particle_Filter
        主粒子滤波器（用于训练）
    DPF_redefined: Differentiable_Particle_Filter
        重定义的粒子滤波器（用于计算ELBO）
    opt: Optimizer
        优化器
    loss: Loss
        损失函数（MSE损失）
    T: int
        时间步数
    data_train: Dataset
        训练数据集
    data_test: Dataset
        测试数据集（可为None）
    batch_size: list[int]
        [训练批量, 验证批量, 测试批量]
    set_fractions: list[float]
        [训练比例, 验证比例, 测试比例]
    epochs: int
        训练轮数
    test_scaling: float
        测试时粒子数放大倍数（如10表示测试时使用10倍粒子）
    opt_schedule: LRScheduler
        学习率调度器
    verbose: bool
        是否打印详细信息
    clip: float
        梯度裁剪值（默认inf表示不裁剪）
    normalise: bool
        是否进行数据标准化
    lam: float
        ELBO损失权重（λ，控制似然损失的比例）
    
    返回：
    ----------
    tuple: (测试损失, 每步损失, 训练时间, 测试时间)
    
    说明：
    - 使用两个粒子滤波器：一个用于训练，一个用于计算ELBO
    - 支持数据标准化，提高训练稳定性
    - 保存验证损失最低的模型
    - 测试时增加粒子数以提高精度
    """
    # 划分数据集
    # 如果 data_test 为 None，从 data_train 划分出验证集和测试集
    # 否则，data_train 作为训练集，data_test 划分出验证集和测试集
    if data_test is None:
        train_set, valid_set, test_set = pt.utils.data.random_split(
            data_train, set_fractions
        )
    else:
        train_set = data_train
        valid_set, test_set = pt.utils.data.random_split(data_test, set_fractions)
    
    print(len(valid_set))
    
    # 设置批量大小
    # -1 表示使用全部数据
    if batch_size[0] == -1:
        batch_size[0] = len(train_set)
    if batch_size[1] == -1:
        batch_size[1] = len(valid_set)
    if batch_size[2] == -1:
        batch_size[2] = len(test_set)

    # 创建数据加载器
    # 训练集：shuffle=True（打乱数据）
    # 验证集和测试集：shuffle=False（不打乱）
    train = pt.utils.data.DataLoader(
        train_set,
        batch_size[0],
        shuffle=True,
        collate_fn=data_train.collate,
        num_workers=data_train.workers,
    )
    valid = pt.utils.data.DataLoader(
        valid_set,
        min(batch_size[1], len(valid_set)),
        shuffle=False,
        collate_fn=data_train.collate,
        num_workers=data_train.workers,
    )
    test = pt.utils.data.DataLoader(
        test_set,
        min(batch_size[2], len(test_set)),
        shuffle=False,
        collate_fn=data_train.collate,
        num_workers=data_train.workers,
        drop_last=True,
    )
    
    # 初始化记录数组
    times = np.empty(epochs)  # 每个epoch的训练时间
    train_loss = np.zeros(len(train) * epochs)  # 所有训练步骤的损失
    test_loss = np.zeros(epochs)  # 每个epoch的验证损失
    min_valid_loss = pt.inf  # 最小验证损失（用于保存最佳模型）

    # 计算数据标准化参数
    # 标准化公式：x' = (x - mean) / sd
    # 计算训练集的均值和标准差
    if normalise:
        for i, simulated_object in enumerate(train):
            if i == 0:
                # 初始化均值和均方值
                # dim=(0, 1): 在批次和时间步维度上求平均
                mean_state = pt.mean(simulated_object.state, dim=(0, 1))
                mean_sq_state = pt.mean(simulated_object.state**2, dim=(0, 1))
                mean_obs = pt.mean(simulated_object.observations, dim=(0, 1))
                mean_sq_obs = pt.mean(simulated_object.observations**2, dim=(0, 1))
            else:
                # 累加均值和均方值
                mean_state += pt.mean(simulated_object.state, dim=(0, 1))
                mean_sq_state += pt.mean(simulated_object.state**2, dim=(0, 1))
                mean_obs += pt.mean(simulated_object.observations, dim=(0, 1))
                mean_sq_obs += pt.mean(simulated_object.observations**2, dim=(0, 1))
        
        # 计算最终均值和标准差
        # 标准差 = sqrt(E[x^2] - E[x]^2)
        mean_state = mean_state / len(train)
        sd_state = pt.sqrt(mean_sq_state / len(train) - mean_state**2)
        mean_obs = mean_obs / len(train)
        sd_obs = pt.sqrt(mean_sq_obs / len(train) - mean_obs**2)

    # 创建复合损失函数（MSE + ELBO）
    # 总损失 = MSE + λ * ELBO
    # λ (lam) 控制似然损失的比例
    if lam != 0:
        # Magnitude_Loss: 计算对数似然因子
        # sign=-1: 最大化似然（最小化负对数似然）
        likelihood_loss = Magnitude_Loss(results.Log_Likelihood_Factors(), sign=-1)
        # Compound_Loss: 组合多个损失函数
        complete_loss = Compound_Loss([loss, likelihood_loss])
    else:
        # 如果 lam=0，只使用 MSE 损失
        complete_loss = loss

    # 训练循环
    for epoch in range(epochs):
        start_ep = time.time()  # 记录epoch开始时间
        DPF.train()  # 设置训练模式
        train_it = enumerate(train)
        
        # 遍历训练批次
        for b, simulated_object in train_it:
            # 如果没有标准化，使用默认值（均值为0，标准差为1）
            if not normalise:
                mean_state = pt.zeros_like(simulated_object.state[0, 0, :])
                mean_obs = pt.ones_like(simulated_object.observations[0, 0, :])
                sd_state = pt.ones_like(simulated_object.state[0, 0, :])
                sd_obs = pt.ones_like(simulated_object.observations[0, 0, :])
            
            # 梯度清零
            # 防止梯度累积，确保每次迭代都是独立的
            opt.zero_grad()
            complete_loss.clear_data()
            
            # 数据标准化
            # 标准化后的数据均值为0，标准差为1，有助于训练稳定性
            if normalise:
                simulated_object.state -= mean_state
                simulated_object.observations -= mean_obs
                simulated_object.state /= sd_state
                simulated_object.observations /= sd_obs

            # 设置状态缩放（用于DIMMPF）
            # 某些模型需要知道数据的缩放参数
            try:
                DPF.model.set_x_scaling(mean_state[0], sd_state[0])
            except:
                pass

            # 计算ELBO损失
            # 使用重定义的粒子滤波器计算对数似然
            if lam != 0:
                likelihood_loss.clear_data()
                # set_up: 预计算似然（需要真实状态和观测）
                DPF_redefined.model.set_up(
                    simulated_object.state[:, :, 0:1], simulated_object.observations
                )
                # 运行重定义的粒子滤波器
                DPF_redefined(simulated_object, T, likelihood_loss.get_reporters())
                likelihood_loss()
                # 设置损失权重：[1.0, lam] 表示 MSE 权重为1，ELBO 权重为λ
                complete_loss.register_data(
                    weights=pt.tensor([1.0, lam], device=DPF.device)
                )
            
            # 注册真实状态
            # 将真实状态传递给损失函数，用于计算预测误差
            loss.register_data(truth=simulated_object)

            # 前向传播
            # 运行主粒子滤波器，生成预测和报告
            DPF(simulated_object, T, complete_loss.get_reporters())

            # 计算损失并反向传播
            complete_loss()  # 计算总损失
            complete_loss.backward()  # 反向传播计算梯度
            
            # 梯度裁剪
            # 防止梯度爆炸，将梯度限制在 [-clip, clip] 范围内
            pt.nn.utils.clip_grad_value_(DPF.parameters(), clip)
            
            # 参数更新
            # 使用优化器根据梯度更新模型参数
            opt.step()

            # 记录训练损失
            # 反标准化损失，使其与原始数据尺度一致
            train_loss[b + len(train) * epoch] = loss.item() * ((sd_state[0]) ** 2)

        # 记录 epoch 时间
        times[epoch] = time.time() - start_ep

        # 学习率调整
        # 根据学习率调度器调整学习率
        if opt_schedule is not None:
            opt_schedule.step()
        
        # 验证
        # 在验证集上评估模型性能
        DPF.eval()  # 设置评估模式
        with pt.inference_mode():  # 禁用梯度计算
            for simulated_object in valid:
                # 数据标准化
                simulated_object.state -= mean_state
                simulated_object.observations -= mean_obs
                simulated_object.state /= sd_state
                simulated_object.observations /= sd_obs
                
                loss.clear_data()
                loss.register_data(truth=simulated_object)
                DPF(simulated_object, T, loss.get_reporters())
                # 累加验证损失
                test_loss[epoch] += loss().item() * ((sd_state[0]) ** 2)

        # 保存最佳模型
        # 如果当前验证损失最低，保存模型参数
        if test_loss[epoch].item() < min_valid_loss:
            min_valid_loss = test_loss[epoch].item()
            best_dict = deepcopy(DPF.state_dict())

        # 打印训练信息
        if verbose:
            print(f"Epoch {epoch}:")
            print(
                f"Train loss: {np.mean(train_loss[epoch * len(train) : (epoch + 1) * len(train)])}"
            )
            print(f"Validation loss: {test_loss[epoch]}\n")

    # 加载最佳模型
    # 使用验证损失最低的模型参数
    DPF.load_state_dict(best_dict)
    
    # 增加测试时的粒子数
    # 测试时使用更多粒子可以提高精度（但速度更慢）
    DPF.n_particles *= test_scaling
    DPF.ESS_threshold *= test_scaling

    # 测试
    start_test = time.time()
    results_ = _test(DPF, loss, T, test, (mean_state, sd_state, mean_obs, sd_obs))
    return *results_, times, np.array([time.time() - start_test])


# =============================================================================
# 第三部分：序列到序列训练函数
# 用于 LSTM 和 Transformer 等神经网络模型
# 特点：直接映射观测序列到状态序列
# =============================================================================

def train_s2s(
    NN: pt.nn.Module,
    opt: pt.optim.Optimizer,
    data: pt.utils.data.Dataset,
    batch_size: Iterable[int],
    set_fractions: Iterable[float],
    epochs: int,
    opt_schedule: pt.optim.lr_scheduler.LRScheduler = None,
    verbose: bool = True,
    clip: float = pt.inf,
):
    """
    序列到序列训练函数（适用于LSTM和Transformer模型）
    
    训练流程：
    1. 划分数据集（训练/验证/测试）
    2. 训练循环：
       - 前向传播：observations -> NN -> predicted_states
       - 计算MSE损失：mean((predicted - true)^2)
       - 反向传播和参数更新
    3. 验证和保存最佳模型
    4. 测试
    
    与 e2e_train 的区别：
    - e2e_train: 使用粒子滤波，需要模拟状态转移和观测过程
    - train_s2s: 直接神经网络映射，端到端学习
    
    参数说明：
    ----------
    NN: Module
        神经网络模型（LSTM或Transformer）
    opt: Optimizer
        优化器
    data: Dataset
        数据集
    batch_size: list[int]
        [训练批量, 验证批量, 测试批量]
    set_fractions: list[float]
        [训练比例, 验证比例, 测试比例]
    epochs: int
        训练轮数
    opt_schedule: LRScheduler
        学习率调度器
    verbose: bool
        是否打印详细信息
    clip: float
        梯度裁剪值（默认inf表示不裁剪）
    
    返回：
    ----------
    tuple: (测试损失, 每步损失, 训练时间, 测试时间)
    """
    try:
        # 划分数据集
        # 按比例划分为训练集、验证集、测试集
        train_set, valid_set, test_set = pt.utils.data.random_split(data, set_fractions)
        
        # 设置批量大小
        # -1 表示使用全部数据
        if batch_size[0] == -1:
            batch_size[0] = len(train_set)
        if batch_size[1] == -1:
            batch_size[1] = len(valid_set)
        if batch_size[2] == -1:
            batch_size[2] = len(test_set)

        # 创建数据加载器
        train = pt.utils.data.DataLoader(
            train_set,
            batch_size[0],
            shuffle=True,  # 训练时打乱数据
            collate_fn=data.collate,
            num_workers=data.workers,
        )
        valid = pt.utils.data.DataLoader(
            valid_set,
            min(batch_size[1], len(valid_set)),
            shuffle=False,  # 验证时不打乱
            collate_fn=data.collate,
            num_workers=data.workers,
        )
        test = pt.utils.data.DataLoader(
            test_set,
            min(batch_size[2], len(test_set)),
            shuffle=False,  # 测试时不打乱
            collate_fn=data.collate,
            num_workers=data.workers,
            drop_last=True,  # 丢弃不完整的批次
        )
    except:
        # 如果划分失败，直接使用传入的 set_fractions 作为数据加载器
        train, valid, test = set_fractions
    
    # 初始化记录数组
    train_loss = np.zeros(len(train) * epochs)  # 所有训练步骤的损失
    test_loss = np.zeros(epochs)  # 每个epoch的验证损失
    min_valid_loss = pt.inf  # 最小验证损失
    times = np.empty(epochs)  # 每个epoch的训练时间
    
    # 训练循环
    for epoch in range(epochs):
        start = time.time()  # 记录epoch开始时间
        NN.train()  # 设置训练模式
        train_it = enumerate(train)
        
        # 遍历训练批次
        for b, simulated_object in train_it:
            # 梯度清零
            opt.zero_grad()
            
            # 前向传播
            # NN 直接映射观测到状态
            x = NN(simulated_object.observations)
            
            # 计算损失（MSE）
            # 比较预测状态与真实状态的均方误差
            # simulated_object.state[:, :, 0:1]: 取状态的第一个维度
            loss = pt.mean((x - simulated_object.state[:, :, 0:1]) ** 2)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            pt.nn.utils.clip_grad_value_(NN.parameters(), clip)
            
            # 参数更新
            opt.step()
            
            # 记录训练损失
            train_loss[b + len(train) * epoch] = loss.item()
        
        # 学习率调整
        if opt_schedule is not None:
            opt_schedule.step()
        
        # 验证
        NN.eval()  # 设置评估模式
        for simulated_object in valid:
            x = NN(simulated_object.observations)
            loss = pt.mean((x - simulated_object.state[:, :, 0:1]) ** 2)
            test_loss[epoch] += loss.item()
        test_loss[epoch] /= len(valid)  # 平均验证损失

        # 保存最佳模型
        if test_loss[epoch] < min_valid_loss:
            min_valid_loss = test_loss[epoch]
            best_dict = deepcopy(NN.state_dict())

        # 打印训练信息
        if verbose:
            print(f"Epoch {epoch}:")
            print(
                f"Train loss: {np.mean(train_loss[epoch * len(train) : (epoch + 1) * len(train)])}"
            )
            print(f"Validation loss: {test_loss[epoch]}\n")
        
        # 记录 epoch 时间
        times[epoch] = time.time() - start
    
    # 加载最佳模型
    NN.load_state_dict(best_dict)
    
    # 测试
    start_test = time.time()
    for simulated_object in test:
        x = NN(simulated_object.observations)
        # 计算每步损失（在状态维度上求平均）
        loss = pt.mean((x - simulated_object.state[:, :, 0:1]) ** 2, dim=2)
        loss = loss.to(device="cpu").detach().numpy()
        print(f"Test loss: {np.mean(loss)}")
        return (
            np.array([np.mean(loss)]),  # 平均测试损失
            np.mean(loss, axis=0),  # 每步损失
            times,  # 训练时间
            np.array([time.time() - start_test]),  # 测试时间
        )
