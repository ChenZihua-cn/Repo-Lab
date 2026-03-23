"""
文件名: main_resnet.py
功能描述: 双中子星(BNS)引力波信号检测的ResNet模型模块
    本模块实现了基于ResNet-18架构的深度学习模型，用于从探测器
    噪声背景中识别双中子星旋近引力波信号。

模型架构:
    ResNet-18 (适配版):
    - 输入: (batch, 1, ndet, fs*T) = (batch, 1, 2, 16384)
    - 输出: (batch, 2) - 二分类（噪声/信号）
    
    关键组件:
    1. BasicBlock: ResNet基础残差块
    2. ResNetBNS: 适配的ResNet-18网络
    3. DatasetGenerator: 数据生成器
    4. 训练和评估函数
    5. ROC曲线绘制

作者: Refactored from main.py
日期: 2026

依赖库:
    - torch: PyTorch深度学习框架
    - torchvision: ResNet预训练模型
    - numpy: 数值计算
    - matplotlib: 绘图
    - sklearn: ROC曲线计算

使用示例:
    from main_resnet import ResNetBNS, DatasetGenerator, train, plot_roc_curve
    
    # 创建模型
    net = ResNetBNS(num_classes=2)
    
    # 创建数据生成器
    dataset = DatasetGenerator(snr=20, nsample_perepoch=100)
    
    # 训练
    train(net, lr=0.003, ...)

参考:
    He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
"""

import os
import numpy as np
from pathlib import Path
from data_prep_bns import sim_data  # 导入BNS数据生成
from utils import Accumulator, Animator, Timer  # 导入工具类

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

# 导入用于ROC绘制的库
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# ResNet基础组件定义
# ============================================================================

class BasicBlock(nn.Module):
    """
    ResNet基础残差块 (BasicBlock)。
    
    这是ResNet架构的核心组件，通过跳跃连接(skip connection)实现
    残差学习，有效缓解深度网络的梯度消失问题。
    
    结构:
        Input ──┬──> Conv1 ──> BN ──> ReLU ──> Conv2 ──> BN ──┬──> (+) ──> ReLU ──> Output
                │                                            │
                └──────────────────(shortcut)────────────────┘
    
    Attributes:
        expansion (int): 输出通道扩展系数，BasicBlock为1
        conv1 (nn.Conv2d): 第一个3x3卷积层
        bn1 (nn.BatchNorm2d): 第一个批归一化层
        conv2 (nn.Conv2d): 第二个3x3卷积层
        bn2 (nn.BatchNorm2d): 第二个批归一化层
        shortcut (nn.Sequential): 快捷连接（当输入输出维度不匹配时使用）
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 卷积步长，默认为1
        downsample (nn.Module): 下采样模块，用于维度匹配
    """
    
    expansion = 1  # BasicBlock不扩展通道数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层：可能进行下采样(stride=2)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个卷积层：保持尺寸
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接
        self.downsample = downsample
        
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入特征图，形状 (N, C, H, W)
        
        Returns:
            Tensor: 输出特征图，形状 (N, out_channels, H', W')
        """
        identity = x  # 保存输入用于残差连接

        # 第一个卷积-BN-ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积-BN
        out = self.conv2(out)
        out = self.bn2(out)

        # 快捷连接（处理维度不匹配）
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加和激活
        out += identity
        out = self.relu(out)

        return out


class ResNetBNS(nn.Module):
    """
    适配双中子星引力波信号检测的ResNet-18模型。
    
    针对引力波时序数据的特点进行适配：
    1. 输入是1D时序数据（2个探测器，16384个时间点）
    2. 使用2D卷积沿时间轴提取特征
    3. 通道数从1逐渐增加到512
    4. 最终输出2类（噪声/信号）
    
    网络结构:
        Conv1 (1->64, k=(2,7), s=(1,2))
        -> MaxPool (k=(1,3), s=(1,2))
        -> Layer1 [2 blocks] (64->64)
        -> Layer2 [2 blocks] (64->128, stride=2)
        -> Layer3 [2 blocks] (128->256, stride=2)
        -> Layer4 [2 blocks] (256->512, stride=2)
        -> AvgPool
        -> FC (512->2)
    
    Attributes:
        conv1 (nn.Conv2d): 初始卷积层
        bn1 (nn.BatchNorm2d): 初始批归一化
        maxpool (nn.MaxPool2d): 初始池化
        layer1-4 (nn.Sequential): 4个残差层
        avgpool (nn.AdaptiveAvgPool2d): 全局平均池化
        fc (nn.Linear): 全连接分类层
    
    Args:
        num_classes (int): 分类类别数，默认为2（噪声/信号）
    """

    def __init__(self, num_classes=2):
        super(ResNetBNS, self).__init__()
        
        self.in_channels = 64  # 当前通道数，用于构建残差层

        # 初始卷积层：沿时间轴的大卷积核
        # 输入: (N, 1, 2, 16384) -> 输出: (N, 64, 1, 8192)
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=(2, 7), stride=(1, 2),
            padding=(0, 3), bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始池化：沿时间轴
        # 输出: (N, 64, 1, 4096)
        self.maxpool = nn.MaxPool2d(
            kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
        )

        # 4个残差层
        # layer1: 64通道，尺寸不变
        self.layer1 = self._make_layer(64, 2, stride=1)
        # layer2: 128通道，时间维度减半
        self.layer2 = self._make_layer(128, 2, stride=(1, 2))
        # layer3: 256通道，时间维度减半
        self.layer3 = self._make_layer(256, 2, stride=(1, 2))
        # layer4: 512通道，时间维度减半
        self.layer4 = self._make_layer(512, 2, stride=(1, 2))

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride=1):
        """
        构建残差层。
        
        Args:
            out_channels (int): 输出通道数
            num_blocks (int): 残差块数量
            stride (int/tuple): 第一个块的步长
        
        Returns:
            nn.Sequential: 残差层
        """
        downsample = None
        
        # 当输入输出维度不匹配时，需要下采样
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * BasicBlock.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        # 第一个块可能进行下采样
        layers.append(
            BasicBlock(self.in_channels, out_channels, stride, downsample)
        )
        self.in_channels = out_channels * BasicBlock.expansion
        
        # 后续块保持尺寸不变
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """He初始化权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播。
        
        Args:
            x (Tensor): 输入数据，形状 (N, 1, 2, 16384)
                        N: batch size
                        1: 通道数（单通道时序）
                        2: 探测器数量（H1, L1）
                        16384: 时间样本数 (8192*2，因为safe=2)
        
        Returns:
            Tensor: 分类logits，形状 (N, num_classes)
        """
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局池化和分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ============================================================================
# 数据集生成器
# ============================================================================

class DatasetGenerator(Dataset):
    """
    双中子星引力波信号数据集生成器。
    
    动态生成训练和测试数据，每个epoch可以生成新的噪声实现，
    有效扩充训练数据量。
    
    Attributes:
        fs (int): 采样频率(Hz)
        T (float): 观测时间(秒)
        detectors (list): 探测器列表
        snr (float): 信噪比
        strains (numpy.ndarray): 生成的应变数据
        labels (numpy.ndarray): 标签（0=噪声，1=信号）
    
    Args:
        fs (int): 采样频率，默认8192 Hz
        T (float): 观测时间，默认1秒
        snr (float): 信噪比，默认20
        detectors (list): 探测器列表，默认['H1', 'L1']
        nsample_perepoch (int): 每个epoch的样本数
        Nnoise (int): 每个信号的噪声实现数
        mdist (str): 质量分布类型
        beta (list): 峰值位置参数
        verbose (bool): 是否打印信息
    """

    def __init__(self, fs=8192, T=1, snr=20,
                 detectors=['H1', 'L1'],
                 nsample_perepoch=100,
                 Nnoise=25, mdist='metric', beta=[0.75, 0.95],
                 verbose=True):
        super(DatasetGenerator, self).__init__()
        
        if verbose:
            print('GPU可用?', torch.cuda.is_available())
        
        self.fs = fs
        self.T = T
        
        # 安全倍数扩展时间窗口
        safe = 2
        self.T *= safe
        
        self.detectors = detectors
        self.snr = snr
        
        # 预生成样本
        self.generate(nsample_perepoch, Nnoise, mdist, beta)

    def generate(self, Nblock, Nnoise=25, mdist='metric', beta=[0.75, 0.95]):
        """
        生成数据。
        
        Args:
            Nblock (int): 样本块大小
            Nnoise (int): 每个信号的噪声实现数
            mdist (str): 质量分布类型
            beta (list): 峰值位置参数
        """
        # 调用BNS数据生成函数
        ts, par = sim_data(
            self.fs, self.T, self.snr, self.detectors,
            Nnoise, size=Nblock, mdist=mdist,
            beta=beta, verbose=False
        )
        
        # 添加通道维度: (N, ndet, T*fs) -> (N, 1, ndet, T*fs)
        self.strains = np.expand_dims(ts[0], 1)
        self.labels = ts[1]

    def __len__(self):
        """返回数据集大小。"""
        return len(self.strains)

    def __getitem__(self, idx):
        """
        获取单个样本。
        
        Args:
            idx (int): 样本索引
        
        Returns:
            tuple: (strain, label)
        """
        return self.strains[idx], self.labels[idx]


# ============================================================================
# 模型加载和保存
# ============================================================================

def load_model(checkpoint_dir=None):
    """
    加载ResNet模型。
    
    从检查点目录加载预训练模型，如果没有则初始化新模型。
    
    Args:
        checkpoint_dir (str): 检查点目录路径
    
    Returns:
        tuple: (net, epoch, train_loss_history)
            - net (ResNetBNS): 模型
            - epoch (int): 已训练轮数
            - train_loss_history (list): 训练损失历史
    """
    net = ResNetBNS(num_classes=2)

    if (checkpoint_dir is not None) and (Path(checkpoint_dir).is_dir()):
        p = Path(checkpoint_dir)
        files = [f for f in os.listdir(p) if '.pt' in f]

        # 加载最新的.pt文件
        if (files != []) and (len(files) >= 1):
            # 按修改时间排序，加载最新的
            files.sort(key=lambda x: os.path.getmtime(p / x), reverse=True)
            checkpoint = torch.load(p / files[0])
            net.load_state_dict(checkpoint['model_state_dict'])
            print('从 {} 加载网络'.format(p / files[0]))
            
            epoch = checkpoint['epoch']
            train_loss_history = np.load(
                p / 'train_loss_history_resnet.npy'
            ).tolist()
            return net, epoch, train_loss_history
        else:
            print('检查点目录中没有.pt文件，初始化新网络')
            return net, 0, []
    else:
        print('初始化新网络!')
        return net, 0, []


def save_model(epoch, model, optimizer, scheduler, checkpoint_dir,
               train_loss_history, filename):
    """
    保存模型检查点。
    
    保存模型参数、优化器状态、学习率调度器状态和训练历史。
    
    Args:
        epoch (int): 当前轮数
        model (nn.Module): 模型
        optimizer (Optimizer): 优化器
        scheduler (_LRScheduler): 学习率调度器
        checkpoint_dir (str): 检查点目录
        train_loss_history (list): 训练损失历史
        filename (str): 文件名
    """
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)

    # 清除旧的.pt文件
    assert '.pt' in filename
    for f in [f for f in os.listdir(p) if '.pt' in f]:
        os.remove(p / f)

    # 保存损失历史
    np.save(p / 'train_loss_history_resnet', train_loss_history)

    # 准备保存内容
    output = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }

    if scheduler is not None:
        output['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存模型
    torch.save(output, p / filename)
    print(f'模型已保存到 {p / filename}')


# ============================================================================
# 训练辅助函数
# ============================================================================

# PyTorch张量操作别名（NumPy风格）
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)


def accuracy(y_hat, y):
    """
    计算预测正确的数量。
    
    Args:
        y_hat (Tensor): 模型预测输出，形状 (N, C) 或 (N,)
        y (Tensor): 真实标签，形状 (N,)
    
    Returns:
        float: 正确预测的数量
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, dim=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


def evaluate_accuracy_gpu(net, data_iter, loss_func, device=None):
    """
    使用GPU计算模型在数据集上的精度。
    
    Args:
        net (nn.Module): 神经网络模型
        data_iter (DataLoader): 数据加载器
        loss_func: 损失函数
        device (torch.device): 计算设备
    
    Returns:
        tuple: (accuracy, loss)
            - accuracy (float): 准确率
            - loss (float): 平均损失
    """
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    
    # 累加器：正确数、总数、损失和
    metric = Accumulator(3)
    
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device).to(torch.float)
            y = y.to(device).to(torch.long)
            y_hat = net(X)
            loss = loss_func(y_hat, y)
            metric.add(accuracy(y_hat, y), y.numel(), loss.sum())
    
    return metric[0] / metric[1], metric[2] / metric[1]


def get_predictions(net, data_iter, device=None):
    """
    获取模型预测概率和真实标签（用于ROC计算）。
    
    Args:
        net (nn.Module): 神经网络模型
        data_iter (DataLoader): 数据加载器
        device (torch.device): 计算设备
    
    Returns:
        tuple: (y_true, y_score)
            - y_true (numpy.ndarray): 真实标签
            - y_score (numpy.ndarray): 预测概率（正类）
    """
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    
    y_true = []
    y_score = []
    
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device).to(torch.float)
            y = y.to(device).to(torch.long)
            
            # 获取模型输出
            logits = net(X)
            
            # 使用softmax获取概率
            probs = torch.softmax(logits, dim=1)
            
            y_true.extend(y.cpu().numpy())
            y_score.extend(probs[:, 1].cpu().numpy())  # 正类概率
    
    return np.array(y_true), np.array(y_score)


# ============================================================================
# ROC曲线绘制
# ============================================================================

def plot_roc_curve(y_true, y_score, save_path=None, title='BNS GW Signal Detection ROC'):
    """
    绘制ROC曲线并计算AUC。
    
    ROC（Receiver Operating Characteristic）曲线是评估二分类模型
    性能的重要工具，展示在不同阈值下的真阳性率(TPR)和假阳性率(FPR)。
    
    AUC（Area Under Curve）是ROC曲线下的面积，取值范围[0, 1]，
    越接近1表示模型性能越好，0.5表示随机猜测。
    
    Args:
        y_true (numpy.ndarray): 真实标签，形状 (N,)
        y_score (numpy.ndarray): 预测正类概率，形状 (N,)
        save_path (str): 图像保存路径，为None则显示图像
        title (str): 图像标题
    
    Returns:
        float: AUC值
    
    Example:
        >>> y_true, y_score = get_predictions(net, test_loader)
        >>> auc = plot_roc_curve(y_true, y_score, save_path='roc.png')
        >>> print(f'AUC = {auc:.4f}')
    """
    # 计算FPR和TPR
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # 计算AUC
    roc_auc = auc(fpr, tpr)
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # 绘制对角线（随机猜测）
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random guess (AUC = 0.5)')
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'ROC曲线已保存到: {save_path}')
    else:
        plt.show()
    
    plt.close()
    
    return roc_auc


def plot_training_history(train_loss_history, save_path=None):
    """
    绘制训练历史曲线。
    
    Args:
        train_loss_history (list): 训练历史，每个元素为[epoch, train_loss, test_loss, train_acc, test_acc]
        save_path (str): 保存路径
    """
    history = np.array(train_loss_history)
    epochs = history[:, 0]
    train_loss = history[:, 1]
    test_loss = history[:, 2]
    train_acc = history[:, 3]
    test_acc = history[:, 4]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss')
    axes[0].plot(epochs, test_loss, 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(epochs, train_acc, 'b-', label='Train Accuracy')
    axes[1].plot(epochs, test_acc, 'r-', label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'训练历史已保存到: {save_path}')
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# 训练函数
# ============================================================================

def train(net, lr, nsample_perepoch, epoch, total_epochs,
          dataset_train, data_loader, test_iter,
          train_loss_history, checkpoint_dir, device, notebook=True):
    """
    训练ResNet模型。
    
    使用Adam优化器和余弦退火学习率调度器进行训练。
    每个epoch生成新的训练样本，并保存最佳模型。
    
    Args:
        net (nn.Module): 神经网络模型
        lr (float): 初始学习率
        nsample_perepoch (int): 每个epoch的样本数
        epoch (int): 起始轮数
        total_epochs (int): 总训练轮数
        dataset_train (DatasetGenerator): 训练数据集生成器
        data_loader (DataLoader): 训练数据加载器
        test_iter (DataLoader): 测试数据加载器
        train_loss_history (list): 训练历史记录
        checkpoint_dir (str): 检查点保存目录
        device (torch.device): 训练设备
        notebook (bool): 是否在Jupyter notebook中运行
    """
    # 定义损失函数和优化器
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )

    # 清空CUDA缓存
    torch.cuda.empty_cache()
    
    # 可视化工具
    if notebook:
        animator = Animator(
            xlabel='epoch', xlim=[1, total_epochs],
            legend=['train loss', 'test loss', 'train acc', 'test acc']
        )
    
    timer, num_batches = Timer(), len(dataset_train)

    # 训练循环
    for epoch in range(epoch, total_epochs):
        # 生成新的训练样本
        dataset_train.generate(nsample_perepoch)

        if not notebook:
            print('学习率: {}'.format(
                optimizer.state_dict()['param_groups'][0]['lr']))

        # 累加器：损失和、准确数和、样本数
        metric = Accumulator(3)

        # 训练模式
        net.train()
        
        for batch_idx, (x, y) in enumerate(data_loader):
            timer.start()
            optimizer.zero_grad()

            # 数据转移到设备
            data = x.to(device, non_blocking=True).to(torch.float)
            label = y.to(device, non_blocking=True).to(torch.long)

            # 前向传播
            pred = net(data)
            loss = loss_func(pred, label)

            # 记录指标
            with torch.no_grad():
                metric.add(loss.sum(), accuracy(pred, label), x.shape[0])
            timer.stop()

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 更新可视化
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if notebook and ((batch_idx + 1) % (num_batches // 5) == 0 or 
                            batch_idx == num_batches - 1):
                animator.add(epoch + (batch_idx + 1) / num_batches,
                           (train_l, None, train_acc, None))

        # 更新学习率
        scheduler.step()

        # 测试集评估
        test_acc, test_l = evaluate_accuracy_gpu(net, test_iter, loss_func, device)

        # 记录历史
        train_loss_history.append([epoch + 1, train_l, test_l, train_acc, test_acc])

        # 更新可视化或打印
        if notebook:
            animator.add(epoch + 1, (train_l, test_l, train_acc, test_acc))
        else:
            print(f'Epoch: {epoch+1} \t'
                  f'Train Loss: {train_l:.4f} Test Loss: {test_l:.4f} \t'
                  f'Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}')

        # 保存模型
        save_model(epoch, net, optimizer, scheduler,
                   checkpoint_dir=checkpoint_dir,
                   train_loss_history=train_loss_history,
                   filename=f'model_e{epoch}.pt')

    # 训练完成总结
    print(f'最终损失 {train_l:.4f}, 训练准确率 {train_acc:.3f}, '
          f'测试准确率 {test_acc:.3f}')
    print(f'{metric[2] * total_epochs / timer.sum():.1f} 样本/秒 '
          f'在 {str(device)}')


# ============================================================================
# 主函数入口
# ============================================================================

if __name__ == "__main__":
    """
    主函数：演示ResNet模型训练流程。
    """
    # 配置参数
    nsample_perepoch = 100
    
    # 创建数据生成器
    print('正在生成数据集...')
    dataset_train = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)
    dataset_test = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)

    # 数据加载器
    data_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_iter = DataLoader(dataset_test, batch_size=32, shuffle=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 检查点目录
    checkpoint_dir = os.path.join(SCRIPT_DIR, 'checkpoints_resnet_bns')

    # 加载或创建模型
    net, epoch, train_loss_history = load_model(checkpoint_dir)
    net.to(device)

    # 训练参数
    lr = 0.001  # ResNet通常使用较小的学习率
    total_epochs = 100
    total_epochs += epoch

    # 开始训练
    print(f'从epoch {epoch}开始训练...')
    train(net, lr, nsample_perepoch, epoch, total_epochs,
          dataset_train, data_loader, test_iter,
          train_loss_history, checkpoint_dir, device, notebook=False)
    
    print('训练完成!')
    
    # 绘制ROC曲线
    print('正在生成ROC曲线...')
    y_true, y_score = get_predictions(net, test_iter, device)
    roc_path = os.path.join(checkpoint_dir, 'roc_curve.png')
    auc_value = plot_roc_curve(y_true, y_score, save_path=roc_path,
                               title='BNS GW Signal Detection - ResNet18')
    print(f'测试集AUC: {auc_value:.4f}')
    
    # 绘制训练历史
    history_path = os.path.join(checkpoint_dir, 'training_history.png')
    plot_training_history(train_loss_history, save_path=history_path)
