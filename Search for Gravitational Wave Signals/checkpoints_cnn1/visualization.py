import os
import numpy as np
import matplotlib.pyplot as plt

# 获取脚本所在目录（checkpoints_cnn1/）的父目录（项目根目录）
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

checkpoint_dir = os.path.join(SCRIPT_DIR, 'checkpoints_cnn1')
history = np.load(os.path.join(checkpoint_dir, 'train_loss_history_cnn.npy'))
# history 形状: [epoch, train_loss, test_loss, train_acc, test_acc]

plt.plot(history[:, 0], history[:, 1], 'o-', label='Train Loss')
plt.plot(history[:, 0], history[:, 2], 's-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(checkpoint_dir, 'training_curve.png'))
plt.show()