#!/usr/bin/env python3
"""
生成比赛提交文件
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from main import MyNet, load_model

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def evaluate_submission(net, data_iter, device):
    """对测试集进行预测"""
    net.eval()
    softmax = torch.nn.Softmax(dim=-1)
    y_hat_list = []
    
    with torch.no_grad():
        for X in data_iter:
            X = X.to(device).to(torch.float)
            y_hat = net(X)
            preds = softmax(y_hat).cpu().numpy()[:, 1].tolist()
            y_hat_list.extend(preds)
    
    return y_hat_list

def main():
    checkpoint_dir = os.path.join(SCRIPT_DIR, 'checkpoints_cnn1')
    test_file = os.path.join(SCRIPT_DIR, 'test.npy')
    output_file = os.path.join(SCRIPT_DIR, 'submission.csv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print('Loading model...')
    net, epoch, _ = load_model(checkpoint_dir)
    net.to(device)
    
    # 加载测试数据
    print(f'Loading test data from {test_file}...')
    test_dataset = np.load(test_file)
    print(f'Test data shape: {test_dataset.shape}')
    
    # 创建 DataLoader（注意没有标签，所以用简单的 tensor dataset）
    data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 预测
    print('Generating predictions...')
    results = evaluate_submission(net, data_loader, device)
    
    # 保存
    submission = pd.DataFrame({
        'id': range(len(results)),
        'target': results
    })
    submission.to_csv(output_file, index=False)
    print(f'Submission saved to {output_file}')

if __name__ == '__main__':
    main()