"""
文件名: evaluate_resnet.py
功能描述: 双中子星引力波信号检测ResNet模型评估脚本
    本脚本用于评估训练好的ResNet模型性能，计算各种评估指标，
    并生成ROC曲线、混淆矩阵等可视化结果。

评估指标:
    - AUC (Area Under ROC Curve): ROC曲线下面积
    - Accuracy: 准确率
    - Precision: 精确率
    - Recall: 召回率
    - F1 Score: F1分数
    - Confusion Matrix: 混淆矩阵

使用方法:
    # 评估最新保存的模型
    python evaluate_resnet.py
    
    # 评估指定检查点目录的模型
    python evaluate_resnet.py --checkpoint_dir ./checkpoints_resnet_bns
    
    # 评估指定epoch的模型
    python evaluate_resnet.py --epoch 50

输出结果:
    - 终端显示各项评估指标
    - roc_curve.png: ROC曲线图
    - confusion_matrix.png: 混淆矩阵图
    - evaluation_report.txt: 详细评估报告

作者: Created for BNS GW detection project
日期: 2026
"""

#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 导入项目模块
from main_resnet import (
    ResNetBNS,
    DatasetGenerator,
    load_model,
    get_predictions
)


# 获取脚本目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_arguments():
    """
    解析命令行参数。
    
    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='评估双中子星引力波信号检测ResNet模型'
    )
    
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=os.path.join(SCRIPT_DIR, 'checkpoints_resnet_bns'),
        help='模型检查点目录路径 (默认: ./checkpoints_resnet_bns)'
    )
    
    parser.add_argument(
        '--epoch',
        type=int,
        default=None,
        help='指定评估的epoch，None则使用最新的模型'
    )
    
    parser.add_argument(
        '--nsamples',
        type=int,
        default=500,
        help='评估样本数 (默认: 500)'
    )
    
    parser.add_argument(
        '--snr',
        type=float,
        default=20,
        help='信噪比 (默认: 20)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批次大小 (默认: 32)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='输出目录，默认使用检查点目录'
    )
    
    return parser.parse_args()


def plot_roc_curve(y_true, y_score, save_path, title='ROC Curve'):
    """
    绘制ROC曲线。
    
    Args:
        y_true (numpy.ndarray): 真实标签
        y_score (numpy.ndarray): 预测概率
        save_path (str): 保存路径
        title (str): 图表标题
    
    Returns:
        float: AUC值
    """
    # 计算FPR, TPR
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # 寻找最佳阈值（约登指数最大）
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    # 标记最佳阈值点
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
             label=f'Optimal threshold = {optimal_threshold:.3f}')
    
    # 对角线
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier (AUC = 0.5)')
    
    # 图形设置
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加信息文本
    info_text = f'Samples: {len(y_true)}\nAUC: {roc_auc:.4f}'
    plt.text(0.6, 0.2, info_text, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'ROC曲线已保存: {save_path}')
    plt.close()
    
    return roc_auc, optimal_threshold


def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['Noise', 'Signal']):
    """
    绘制混淆矩阵。
    
    Args:
        y_true (numpy.ndarray): 真实标签
        y_pred (numpy.ndarray): 预测标签
        save_path (str): 保存路径
        class_names (list): 类别名称
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始数值
    im1 = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(class_names)
    axes[0].set_yticklabels(class_names)
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight='bold')
    
    plt.colorbar(im1, ax=axes[0])
    
    # 百分比
    im2 = axes[1].imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title('Confusion Matrix (Percent)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(class_names)
    axes[1].set_yticklabels(class_names)
    
    # 添加百分比标注
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{cm_percent[i, j]:.1f}%',
                        ha="center", va="center",
                        color="white" if cm_percent[i, j] > 50 else "black",
                        fontsize=14, fontweight='bold')
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'混淆矩阵已保存: {save_path}')
    plt.close()
    
    return cm


def plot_score_distribution(y_true, y_score, save_path, threshold=0.5):
    """
    绘制预测分数分布。
    
    Args:
        y_true (numpy.ndarray): 真实标签
        y_score (numpy.ndarray): 预测概率
        save_path (str): 保存路径
        threshold (float): 分类阈值
    """
    plt.figure(figsize=(10, 6))
    
    # 分离正负样本的分数
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    
    # 绘制直方图
    plt.hist(neg_scores, bins=50, alpha=0.6, label='Noise (Class 0)',
             color='blue', edgecolor='black', linewidth=0.5)
    plt.hist(pos_scores, bins=50, alpha=0.6, label='Signal (Class 1)',
             color='red', edgecolor='black', linewidth=0.5)
    
    # 标记阈值线
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.3f}')
    
    plt.xlabel('Predicted Probability (Signal Class)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'分数分布图已保存: {save_path}')
    plt.close()


def evaluate_model(net, test_iter, device, threshold=0.5):
    """
    全面评估模型性能。
    
    Args:
        net (nn.Module): 神经网络模型
        test_iter (DataLoader): 测试数据加载器
        device (torch.device): 计算设备
        threshold (float): 分类阈值
    
    Returns:
        dict: 包含各项评估指标的字典
    """
    # 获取预测
    y_true, y_score = get_predictions(net, test_iter, device)
    
    # 根据阈值获取预测标签
    y_pred = (y_score >= threshold).astype(int)
    
    # 计算各项指标
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'y_true': y_true,
        'y_score': y_score,
        'y_pred': y_pred
    }
    
    return metrics


def print_evaluation_report(metrics, threshold, output_file=None):
    """
    打印并保存评估报告。
    
    Args:
        metrics (dict): 评估指标字典
        threshold (float): 分类阈值
        output_file (str): 输出文件路径
    """
    report = []
    report.append('=' * 60)
    report.append('双中子星引力波信号检测 - 模型评估报告')
    report.append('=' * 60)
    report.append('')
    report.append('【分类阈值】')
    report.append(f'  Threshold = {threshold:.4f}')
    report.append('')
    report.append('【评估指标】')
    report.append(f'  Accuracy  (准确率):  {metrics["accuracy"]:.4f}')
    report.append(f'  Precision (精确率):  {metrics["precision"]:.4f}')
    report.append(f'  Recall    (召回率):  {metrics["recall"]:.4f}')
    report.append(f'  F1 Score  (F1分数):  {metrics["f1"]:.4f}')
    report.append('')
    report.append('【样本统计】')
    report.append(f'  总样本数: {len(metrics["y_true"])}')
    report.append(f'  正样本数 (Signal): {sum(metrics["y_true"])}')
    report.append(f'  负样本数 (Noise):  {len(metrics["y_true"]) - sum(metrics["y_true"])}')
    report.append('')
    
    # 混淆矩阵
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    report.append('【混淆矩阵】')
    report.append(f'                 预测')
    report.append(f'              Noise  Signal')
    report.append(f'  真实 Noise  {cm[0,0]:4d}   {cm[0,1]:4d}')
    report.append(f'       Signal {cm[1,0]:4d}   {cm[1,1]:4d}')
    report.append('')
    report.append('=' * 60)
    
    # 打印到终端
    for line in report:
        print(line)
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        print(f'\n评估报告已保存: {output_file}')


def main():
    """
    主评估函数。
    
    Returns:
        int: 0表示成功
    """
    # 解析参数
    args = parse_arguments()
    
    print('=' * 60)
    print('双中子星引力波信号检测 - ResNet模型评估')
    print('=' * 60)
    print()
    
    # 确定输出目录
    output_dir = args.output_dir if args.output_dir else args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # ========================================================================
    # 设备配置
    # ========================================================================
    
    print('【步骤1/5】配置计算设备...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  使用设备: {device}')
    print()
    
    # ========================================================================
    # 模型加载
    # ========================================================================
    
    print('【步骤2/5】加载模型...')
    
    # 如果指定了epoch，尝试加载该epoch的模型
    if args.epoch is not None:
        model_path = Path(args.checkpoint_dir) / f'model_e{args.epoch}.pt'
        if model_path.exists():
            print(f'  加载指定epoch模型: {model_path}')
            checkpoint = torch.load(model_path, map_location=device)
            net = ResNetBNS(num_classes=2)
            net.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', args.epoch)
        else:
            print(f'  警告: 未找到epoch {args.epoch}的模型，使用最新模型')
            net, start_epoch, _ = load_model(args.checkpoint_dir)
    else:
        net, start_epoch, _ = load_model(args.checkpoint_dir)
    
    net.to(device)
    net.eval()
    print(f'  模型已加载 (epoch {start_epoch})')
    print()
    
    # ========================================================================
    # 数据生成
    # ========================================================================
    
    print('【步骤3/5】生成测试数据...')
    print(f'  样本数: {args.nsamples}')
    print(f'  信噪比: {args.snr}')
    
    dataset_test = DatasetGenerator(
        fs=8192,
        T=1,
        snr=args.snr,
        detectors=['H1', 'L1'],
        nsample_perepoch=args.nsamples,
        Nnoise=25,
        mdist='metric',
        beta=[0.75, 0.95],
        verbose=False
    )
    
    test_iter = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print('  测试数据生成完成')
    print()
    
    # ========================================================================
    # 模型评估
    # ========================================================================
    
    print('【步骤4/5】评估模型性能...')
    
    # 先用默认阈值0.5评估，获取AUC和最佳阈值
    y_true, y_score = get_predictions(net, test_iter, device)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f'  AUC = {roc_auc:.4f}')
    print(f'  最佳阈值 = {optimal_threshold:.4f}')
    
    # 使用最佳阈值进行全面评估
    metrics = evaluate_model(net, test_iter, device, threshold=optimal_threshold)
    metrics['auc'] = roc_auc
    
    print('  评估完成')
    print()
    
    # ========================================================================
    # 结果输出和可视化
    # ========================================================================
    
    print('【步骤5/5】生成可视化结果...')
    
    # 打印评估报告
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    print_evaluation_report(metrics, optimal_threshold, report_path)
    
    # 绘制ROC曲线
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plot_roc_curve(
        metrics['y_true'],
        metrics['y_score'],
        roc_path,
        title=f'BNS GW Detection - ResNet18 (AUC={roc_auc:.4f})'
    )
    
    # 绘制混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(
        metrics['y_true'],
        metrics['y_pred'],
        cm_path
    )
    
    # 绘制分数分布
    dist_path = os.path.join(output_dir, 'score_distribution.png')
    plot_score_distribution(
        metrics['y_true'],
        metrics['y_score'],
        dist_path,
        threshold=optimal_threshold
    )
    
    print()
    print('=' * 60)
    print('评估全部完成！')
    print(f'结果保存在: {output_dir}')
    print('=' * 60)
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print('\n\n评估被用户中断')
        sys.exit(1)
    except Exception as e:
        print(f'\n\n评估过程出错: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
