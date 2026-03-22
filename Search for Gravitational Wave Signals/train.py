#!/usr/bin/env python3
"""
main.py里已经生成了预训练模型
这个文件整理了训练引力波检测模型的过程
对应原 notebook 的 Train 部分
"""
import os
import torch
from torch.utils.data import DataLoader
from main import DatasetGenerator, MyNet, load_model, save_model, train

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # 配置参数
    # 初始化数据生成 class
    nsample_perepoch = 100  # 每个 epoch 的样本数
    
    # 数据集
    print('Generating datasets...')
    dataset_train = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)
    dataset_test = DatasetGenerator(snr=20, nsample_perepoch=nsample_perepoch)

    # 创建一个DataLoader
    data_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
    test_iter = DataLoader(dataset_test, batch_size=32, shuffle=True)
    
    # 设备使用CUDA设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 模型和损失历史的输出路径（基于脚本位置）
    checkpoint_dir = os.path.join(SCRIPT_DIR, 'checkpoints_cnn1')
 
    # 创建模型
    net, epoch, train_loss_history = load_model(checkpoint_dir)
    net.to(device)
    
    lr = 0.003 # 学习率
    total_epochs = 100 # 总的训练轮数
    total_epochs += epoch  # 加上已经训练过的轮数
    output_freq = 1  # 输出频率
    snr = 20 # 信噪比 (SNR): 默认设置为 20
    batch_size=32 # 批次大小

    # 训练
    print(f'Starting training from epoch {epoch}...')
    train(net, lr, nsample_perepoch, epoch, total_epochs,
          dataset_train, data_loader, test_iter,
          train_loss_history, checkpoint_dir, device, notebook=False)
    
    print('Training completed!')

if __name__ == '__main__':
    main()