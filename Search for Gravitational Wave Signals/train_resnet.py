"""
文件名: train_resnet.py
功能描述: 双中子星引力波信号检测ResNet模型训练脚本
    本脚本整理并执行双中子星(BNS)引力波信号检测的ResNet模型训练流程，
    对应原notebook的Train部分。

训练流程:
    1. 配置训练参数（学习率、批次大小、epoch数等）
    2. 初始化BNS数据生成器
    3. 加载或创建ResNet-18模型
    4. 执行训练循环
    5. 保存训练好的模型
    6. 生成ROC曲线和训练历史图

命令行运行:
    python train_resnet.py

或从其他脚本导入:
    from train_resnet import main
    main()

依赖库:
    - torch: PyTorch深度学习框架
    - main_resnet: ResNet模型和数据生成器模块

作者: Refactored from train.py
日期: 2026
"""

#!/usr/bin/env python3

import os
import sys

# 导入ResNet模型相关组件
from main_resnet import (
    DatasetGenerator,      # BNS数据生成器
    ResNetBNS,             # ResNet模型
    load_model,            # 模型加载函数
    save_model,            # 模型保存函数
    train,                 # 训练函数
    get_predictions,       # 获取预测函数
    plot_roc_curve,        # ROC曲线绘制
    plot_training_history  # 训练历史绘制
)

import torch
from torch.utils.data import DataLoader


# 获取脚本所在目录，用于确定相对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    """
    主训练函数。
    
    执行完整的ResNet模型训练流程，包括数据生成、模型训练、
    结果可视化和模型保存。
    
    Returns:
        int: 0表示成功完成
    """
    # ========================================================================
    # 配置参数
    # ========================================================================
    
    # 数据生成参数
    nsample_perepoch = 100  # 每个epoch生成的样本数
    # 总样本数 = nsample_perepoch，其中一半是噪声，一半是信号
    # 信号样本通过Nnoise个噪声实现来增强
    
    # 信噪比(SNR)配置
    snr = 20  # 信号信噪比，控制信号强度
    # SNR越高，信号越容易检测；SNR越低，任务越有挑战性
    
    # 数据加载参数
    batch_size = 32  # 批次大小
    # 较大的batch_size可以提高训练速度，但需要更多显存
    # 较小的batch_size可以提供更稳定的梯度估计
    
    # 训练参数
    lr = 0.001  # 初始学习率
    # ResNet通常使用较小的学习率（比CNN的0.003更小）
    # 使用余弦退火调度器动态调整学习率
    
    total_epochs = 100  # 总训练轮数
    # 每轮包含:
    # 1. 生成新的训练样本
    # 2. 完整遍历训练数据
    # 3. 测试集评估
    # 4. 模型保存
    
    # ========================================================================
    # 数据集初始化
    # ========================================================================
    
    print('=' * 60)
    print('双中子星引力波信号检测 - ResNet模型训练')
    print('=' * 60)
    print()
    
    print('【步骤1/6】正在生成数据集...')
    print(f'  - 训练集: 每epoch {nsample_perepoch} 样本')
    print(f'  - 测试集: 每epoch {nsample_perepoch} 样本')
    print(f'  - 信噪比: SNR = {snr}')
    
    # 创建训练数据集生成器
    # DatasetGenerator会动态生成BNS信号和噪声数据
    dataset_train = DatasetGenerator(
        fs=8192,               # 采样频率 8192 Hz
        T=1,                   # 观测时间 1 秒
        snr=snr,               # 信噪比
        detectors=['H1', 'L1'], # LIGO Hanford和Livingston
        nsample_perepoch=nsample_perepoch,
        Nnoise=25,             # 每个信号25个噪声实现
        mdist='metric',        # 度规质量分布
        beta=[0.75, 0.95],     # 峰值位置参数
        verbose=True
    )
    
    # 创建测试数据集生成器
    dataset_test = DatasetGenerator(
        fs=8192,
        T=1,
        snr=snr,
        detectors=['H1', 'L1'],
        nsample_perepoch=nsample_perepoch,
        Nnoise=25,
        mdist='metric',
        beta=[0.75, 0.95],
        verbose=True
    )
    
    print('数据集生成完成！\n')

    # ========================================================================
    # 数据加载器创建
    # ========================================================================
    
    print('【步骤2/6】正在创建数据加载器...')
    print(f'  - 批次大小: {batch_size}')
    print(f'  - 训练批次: {len(dataset_train) // batch_size}')
    print(f'  - 测试批次: {len(dataset_test) // batch_size}')
    
    # DataLoader负责批量加载和打乱数据
    data_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,          # 每个epoch打乱数据
        num_workers=0,         # 数据加载工作进程数
        pin_memory=True        # 加速GPU数据传输
    )
    
    test_iter = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    print('数据加载器创建完成！\n')

    # ========================================================================
    # 设备配置
    # ========================================================================
    
    print('【步骤3/6】正在配置计算设备...')
    
    # 自动检测并选择设备（优先GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  - 使用设备: {device}')
    
    if device.type == 'cuda':
        # 显示GPU信息
        print(f'  - GPU型号: {torch.cuda.get_device_name(0)}')
        print(f'  - 显存分配: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    else:
        print('  - 警告: 未检测到GPU，将使用CPU训练（速度较慢）')
    
    print()

    # ========================================================================
    # 模型加载/创建
    # ========================================================================
    
    print('【步骤4/6】正在加载/创建模型...')
    
    # 检查点保存目录
    checkpoint_dir = os.path.join(SCRIPT_DIR, 'checkpoints_resnet_bns')
    print(f'  - 检查点目录: {checkpoint_dir}')
    
    # 加载现有模型或创建新模型
    net, start_epoch, train_loss_history = load_model(checkpoint_dir)
    
    # 将模型转移到设备
    net.to(device)
    
    # 显示模型信息
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'  - 模型类型: ResNet-18 (适配BNS检测)')
    print(f'  - 总参数数: {total_params:,}')
    print(f'  - 可训练参数: {trainable_params:,}')
    print(f'  - 起始轮数: {start_epoch}')
    print(f'  - 目标轮数: {start_epoch + total_epochs}')
    print()

    # ========================================================================
    # 训练执行
    # ========================================================================
    
    print('【步骤5/6】开始训练...')
    print(f'  - 初始学习率: {lr}')
    print(f'  - 学习率调度: CosineAnnealingLR')
    print(f'  - 优化器: Adam')
    print(f'  - 损失函数: CrossEntropyLoss')
    print()
    print('-' * 60)
    
    # 计算实际总epoch数（加上已训练的）
    target_epoch = total_epochs + start_epoch
    
    # 执行训练
    train(
        net=net,
        lr=lr,
        nsample_perepoch=nsample_perepoch,
        epoch=start_epoch,
        total_epochs=target_epoch,
        dataset_train=dataset_train,
        data_loader=data_loader,
        test_iter=test_iter,
        train_loss_history=train_loss_history,
        checkpoint_dir=checkpoint_dir,
        device=device,
        notebook=False  # 命令行模式
    )
    
    print('-' * 60)
    print('训练阶段完成！\n')

    # ========================================================================
    # 结果可视化和保存
    # ========================================================================
    
    print('【步骤6/6】生成可视化结果...')
    
    # 重新生成测试数据用于最终评估
    print('  - 重新生成测试样本用于ROC评估...')
    dataset_test.generate(nsample_perepoch)
    test_iter = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # 获取预测结果
    print('  - 获取模型预测...')
    y_true, y_score = get_predictions(net, test_iter, device)
    
    # 绘制ROC曲线
    print('  - 绘制ROC曲线...')
    roc_path = os.path.join(checkpoint_dir, 'roc_curve_final.png')
    auc_value = plot_roc_curve(
        y_true=y_true,
        y_score=y_score,
        save_path=roc_path,
        title='BNS GW Signal Detection - ResNet18 ROC Curve'
    )
    print(f'  - 测试集AUC: {auc_value:.4f}')
    
    # 绘制训练历史
    print('  - 绘制训练历史...')
    history_path = os.path.join(checkpoint_dir, 'training_history.png')
    plot_training_history(train_loss_history, save_path=history_path)
    
    print()
    print('=' * 60)
    print('训练全部完成！')
    print(f'模型和结果保存在: {checkpoint_dir}')
    print('=' * 60)
    
    return 0


if __name__ == '__main__':
    # 程序入口
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print('\n\n训练被用户中断')
        sys.exit(1)
    except Exception as e:
        print(f'\n\n训练过程出错: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
