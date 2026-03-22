#!/usr/bin/env python3
"""
评估模型性能，绘制 ROC 曲线
对应原 notebook 的 Evaluate 部分
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from main import DatasetGenerator, MyNet, load_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def evaluate_model(net, data_iter, device):
    """评估模型，返回预测概率和真实标签"""
    net.eval()
    softmax = torch.nn.Softmax(dim=-1)
    y_hat_list, y_list = [], []
    
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device).to(torch.float)
            y = y.to(device).to(torch.long)
            y_hat = net(X)
            
            preds = softmax(y_hat).cpu().numpy()[:, 1].tolist()
            labels = y.cpu().numpy().tolist()
            
            y_hat_list.extend(preds)
            y_list.extend(labels)
    
    return np.asarray(y_hat_list), np.asarray(y_list)

def main():
    # 配置
    nsample_perepoch = 1000
    snr = 20
    batch_size = 32
    checkpoint_dir = os.path.join(SCRIPT_DIR, 'checkpoints_cnn1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print('Loading model...')
    net, epoch, _ = load_model(checkpoint_dir)
    net.to(device)
    print(f'Model loaded from epoch {epoch}')
    
    # 生成测试数据
    print('Generating test data...')
    dataset_test = DatasetGenerator(snr=snr, nsample_perepoch=nsample_perepoch)
    data_iter = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # 评估
    print('Evaluating...')
    y_hat_list, y_list = evaluate_model(net, data_iter, device)
    
    # 计算 ROC 和 AUC
    fpr, tpr, thresholds = roc_curve(y_list, y_hat_list)
    auc = roc_auc_score(y_list, y_hat_list)
    print(f'AUC: {auc:.6f}')
    
    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='deeppink', label=f'Model (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.50)')
    
    # 标记阈值 0.5 的点
    idx = np.argmax(thresholds < 0.5)
    plt.scatter(fpr[idx], tpr[idx], label=f'Threshold={thresholds[idx]:.2f}')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    roc_path = os.path.join(SCRIPT_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=150)
    print(f'ROC curve saved to {roc_path}')
    plt.show()

if __name__ == '__main__':
    main()