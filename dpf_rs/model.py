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
        内部类：用于重索引数组
        创建后应视为不可变和只读
        设计用途：以 y[t] 的形式访问时间 t 的观测值
        
        参数:
        ----------
        base_index: int
            第一个存储项的期望索引
        
        *args: any
            传递给 np.array() 的参数
        
        **kwargs: any
            传递给 np.array() 的参数
        """

        def __init__(self, base_index: int, ls):
            super().__init__()
            self.array = ls          # 存储的数组
            self.base_index = base_index  # 基础索引偏移量

        def __getitem__(self, index):
            # 通过偏移量访问实际数组
            return self.array[index - self.base_index]

    def set_observations(self, get_observation: Callable, t: int):
        """设置观测值的函数（需要在子类中实现）"""
        NotImplementedError("Function to set observations not implemented")

    def to(self, **kwargs):
        """将模型移动到指定设备"""
        if kwargs["device"] is not None:
            self.device = kwargs["device"]
        for var in vars(self):
            if isinstance(var, pt.Tensor) and not isinstance(var, pt.nn.Parameter):
                var.to(dtype=kwargs["dtype"], device=kwargs["device"])
        super().to(**kwargs)

    def __init__(self, device: str = "cuda") -> None:
        """
        初始化 Feynman-Kac 模型
        
        参数:
        ----------
        device: str
            运行设备，'cuda' 或 'cpu'
        """
        super().__init__()
        self.alg = self.PF_Type.Undefined  # 默认未定义类型
        self.device = device
        # CPU 设备不支持 Generator(device=...)，使用默认生成器
        if device == "cuda":
            self.rng = pt.Generator(device=self.device)
        else:
            self.rng = pt.Generator()

    # 评估 G_0: 时间 0 的权重函数
    def log_G_0(self, x_0):
        """时间 0 的权重函数（对数形式）"""
        NotImplementedError("Weighting function not implemented for time 0")

    # 采样 M_0: 时间 0 的提议分布
    def M_0_proposal(self, batches: int, n_samples: int):
        """时间 0 的提议分布采样"""
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
        """时间 0 的动态/提议 Radon-Nikodym 导数"""
        raise NotImplementedError(
            "Dynamic/Proposal Radon-Nikodym derivative not implemented for time zero"
        )

    def log_R_t(self, x_t, x_t_1, t: int):
        """时间 t 的动态/提议 Radon-Nikodym 导数"""
        raise NotImplementedError(
            "Dynamic/Proposal Radon-Nikodym derivative not implemented for time t"
        )

    def log_f_t(self, x_t, t: int):
        """观测似然函数"""
        raise NotImplementedError("Observation likelihood not implemented")

    def log_eta_t(self, x_t, t: int):
        """辅助权重函数"""
        raise NotImplementedError("Auxililary weights not implemented")

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
        """生成时间 0 的状态"""
        raise NotImplementedError("State generation not implemented for time 0")

    def M_0_proposal(self, batches: int, n_samples: int):
        """
        时间 0 的提议分布采样
        
        工作流程:
        1. 生成候选状态
        2. 计算每个候选的概率
        3. 使用多项式采样选择
        """
        state = self.generate_state_0()  # SxD
        probs = self.log_M_0(state.unsqueeze(0)).squeeze()  # S
        indices = pt.multinomial(pt.exp(probs), batches * n_samples, True).reshape(
            batches, n_samples
        )  # BxN
        return nd_select(state, indices)  # BxNxD

    def generate_state_t(self, x_t_1, t: int):
        """生成时间 t 的状态"""
        raise NotImplementedError("State generation not implemented for time t")

    def M_t_proposal(self, x_t_1, t: int):
        """
        时间 t 的提议分布采样
        
        工作流程:
        1. 基于前一时刻状态生成候选
        2. 计算每个候选的概率
        3. 使用多项式采样选择
        """
        state = self.generate_state_t(x_t_1, t)  # BxNxSxD
        probs = self.log_M_t(state, x_t_1, t)  # BxNxS
        indices = pt.multinomial(pt.flatten(pt.exp(probs), 0, 1), 1, True).reshape(
            x_t_1.shape(0), x_t_1.shape(1)
        )  # BxN
        return batched_select(state, indices)  # BxNxD


# =============================================================================
# 第四部分：状态空间对象
# 用于生成和管理观测数据
# =============================================================================

class State_Space_Object:
    """
    通用状态空间对象的基类
    
    可以生成观测值并更新状态
    真实状态可能不可用，此时返回 NaN
    
    维护一个列表，假设包含连续时间步的观测值
    以及一个索引变量存储列表中第一个值的时间步
    
    每次请求观测值时:
    - 如果存在则返回
    - 如果不存在则推进状态并顺序生成观测值直到达到所需时间
    
    参数:
    -----------
    observation_history_length: int
        任何时间步保持的观测值数量
    
    observation_dimension: int
        观测向量的维度
    
    注意:
    --------
    不建议在创建和 get_observation() 方法之外使用此类
    
    状态转移和返回观测值的随机数生成器是分开的
    因为它们可能以不同顺序执行
    """

    def _get_observation(self, t):
        """获取时间 t 的观测值（需要在子类中实现）"""
        pass

    def save(self):
        """保存数据（需要在子类中实现）"""
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
        
        参数:
        ----------
        t: int
            新观测值的时间步
        
        value: Tensor
            新观测值的值
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
        
        参数:
        ----------
        t: int
            要获取观测值的时间步
        
        返回:
        ----------
        Tensor: 时间 t 的观测值
        """
        if t < 0:
            # 负时间返回 NaN
            return pt.full(
                (self.batch_size, self.observation_dimension),
                pt.nan,
                device=self.device,
            )

        if t < self.time_index:
            raise ValueError(
                f"Trying to access observation at time {t}, "
                f"the earliest stored is at time {self.time_index}"
            )

        if t == 0 and not self.first_object_set:
            # 时间 0 的首次设置
            self.first_object_set = True
            self._set_observation(0, self.model.observation_generation(self.x_t))
            return self.observations[:, 0]

        # 生成直到时间 t 的所有观测值
        while t > self.object_time:
            self._forward()
            self._set_observation(
                self.object_time, self.model.observation_generation(self.x_t)
            )

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
        
        参数:
        ----------
        path: str
            保存路径
        T: int
            每个序列的时间步数
        quantity: int
            要生成的序列数量
        prefix: str
            文件名前缀
        clear_folder: bool
            是否清空目标文件夹
        bypass_ask: bool
            是否跳过确认提示
        """
        # 检查模型类型
        if self.model.alg != self.model.PF_Type.Bootstrap:
            warn(
                f"Model is {self.model.alg.name} instead of Bootstrap, are you this is right?"
            )
        
        # 清空文件夹
        if clear_folder:
            if os.path.exists(path):
                if bypass_ask:
                    response = "Y"
                else:
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

        # 生成并保存数据
        for i in range(quantity):
            temp = copy.copy(self)
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
