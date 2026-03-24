# 引力波信号检测系统技术文档

本文档详细说明引力波信号（BBH/BNS）数据生成方法、深度学习模型架构、模块引用关系及评估指标。

---

## 目录

1. [系统概述](#一系统概述)
2. [数据生成方法](#二数据生成方法)
   - 2.1 [BBH（Binary Black Hole，双黑洞）信号](#21-bbhbinary-black-hole双黑洞信号)
   - 2.2 [BNS（Binary Neutron Star，双中子星）信号](#22-bnsbinary-neutron-star双中子星信号)
3. [模型架构](#三模型架构)
   - 3.1 [CNN（Convolutional Neural Network，卷积神经网络）模型](#31-cnnconvolutional-neural-network卷积神经网络模型)
   - 3.2 [ResNet（Residual Network，残差网络）模型](#32-resnetresidual-network残差网络模型)
4. [模块引用关系](#四模块引用关系)
5. [关键参数配置](#五参数配置)
6. [模型评估指标](#六模型评估指标)
7. [训练基础概念](#七训练基础概念)

---

## 一、系统概述

本系统用于检测引力波（Gravitational Wave，GW）信号，支持两种波源类型：

| 波源类型 | 英文全称 | 质量范围 | 特征频率 | 波形模型 |
|---------|---------|---------|---------|---------|
| **BBH** | Binary Black Hole（双黑洞） | 5-100 M☉ | 较低 (~10-100 Hz) | IMRPhenomD |
| **BNS** | Binary Neutron Star（双中子星） | 2-4 M☉ | 较高 (~100-1000 Hz) | IMRPhenomD |

系统包含数据生成模块 (`data_prep_*.py`)、模型定义 (`main*.py`) 和训练评估脚本 (`train*.py`, `evaluate*.py`)。

---

## 二、数据生成方法

### 2.1 BBH（Binary Black Hole，双黑洞）信号

**文件**: `data_prep_bbh.py`

#### 2.1.1 噪声生成

```python
# 生成探测器噪声的PSD（Power Spectral Density，功率谱密度）
psd = gen_psd(fs, T_obs, op='AdvDesign', det='H1')

# 根据PSD生成时域高斯噪声
noise = gen_noise(fs, T_obs, psd)
# 原理：频域复高斯随机数 → 乘以sqrt(PSD/4) → IFFT（Inverse Fast Fourier Transform，逆快速傅里叶变换）得到时域噪声
```

#### 2.1.2 BBH 信号生成

使用 LALSimulation 的 IMRPhenomD 波形近似：

```python
hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
    m1*MSUN_SI, m2*MSUN_SI,  # 质量 (5-100 M☉)
    0, 0, 0, 0, 0, 0,         # 自旋（设为0）
    1e6*PC_SI,                # 距离（1Mpc）
    iota, phi, 0, 0, 0,       # 倾角、相位
    1/fs, f_low, f_low,       # 采样率、起始频率 (~10Hz)
    approximant=IMRPhenomD
)
```

#### 2.1.3 探测器响应

- **天线响应**：根据天空位置(ra, dec)和极化角(psi)计算 Fp、Fc
- **时间延迟**：计算信号到达不同探测器相对于地心的延迟

#### 2.1.4 数据合成流程 (`sim_data`)

```
┌─────────────────────────────────────────────────────────┐
│  噪声类样本 (label=0):                                   │
│     纯噪声 → 白化 → 保存                                 │
│                                                         │
│  信号类样本 (label=1):                                   │
│     随机生成BBH参数(m1, m2, ra, dec等)                  │
│     生成波形(hp, hc) → 应用探测器响应                    │
│     归一化到目标SNR（Signal-to-Noise Ratio，信噪比）    │
│     每个信号叠加 Nnoise 组不同噪声实现                   │
│     白化处理 → 保存                                      │
└─────────────────────────────────────────────────────────┘
```

#### 2.1.5 质量分布 (`gen_masses`)

| 模式 | 说明 | 质量范围 |
|------|------|---------|
| `astro` | 天体物理对数分布 | 5-100 M☉ |
| `gh` | George & Huerta分布 (q~[1,10]) | 5-100 M☉ |
| `metric` | 度规基础分布 | 5-100 M☉ |

---

### 2.2 BNS（Binary Neutron Star，双中子星）信号

**文件**: `data_prep_bns.py`

#### 2.2.1 BNS 物理参数

| 参数 | 说明 | 范围 |
|------|------|------|
| m1, m2 | 中子星质量 | 1.0-2.0 M☉ |
| M | 总质量 | 2.0-4.0 M☉ |
| mc | 啁啾质量（Chirp Mass） | Mc = (m1*m2)^(3/5) / (m1+m2)^(1/5) |
| eta | 对称质量比 | η = m1*m2 / (m1+m2)^2 |

#### 2.2.2 BNS 信号生成

与BBH类似，但关键区别：

```python
# 起始频率更高（BNS特征频率更高）
f_low = 15.0  # Hz (BBH 约 10Hz)

# 质量范围更小
m1, m2 = 1.0-2.0 M☉  # 单个中子星
M = 2.0-4.0 M☉       # 总质量

# 使用相同的IMRPhenomD波形模型
hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
    par.m1 * lal.MSUN_SI, par.m2 * lal.MSUN_SI,
    0, 0, 0, 0, 0, 0,
    dist, par.iota, par.phi, 0, 0, 0,
    1/fs, f_low, f_low,
    lal.CreateDict(),
    approximant  # IMRPhenomD
)
```

#### 2.2.3 BNS 质量分布

| 模式 | 特点 |
|------|------|
| `astro` | 天体物理对数分布，单个中子星 1.0-2.0 M☉ |
| `gh` | George & Huerta分布，质量比 q~[1, 2] |
| `metric` | 度规基础分布，总质量 2.0-4.0 M☉ |

#### 2.2.4 核心函数

| 函数 | 功能 |
|------|------|
| `gen_par()` | 生成随机BNS参数集 |
| `gen_bns()` | 生成BNS时域信号 |
| `make_bns()` | 应用探测器响应和时间延迟 |
| `sim_data()` | 主数据生成函数 |
| `get_fmin()` | 使用2PN（2-Post-Newtonian，二阶后牛顿）啁啾时间公式计算起始频率 |

---

## 三、模型架构

### 3.1 CNN（Convolutional Neural Network，卷积神经网络）模型

**文件**: `main.py`

#### 3.1.1 网络结构

```
输入: (batch, 1, ndet, fs*T)  [默认: (N, 1, 2, 16384)]

卷积层 1: Conv2d(1→8,  kernel=(1,32)) → ELU → BatchNorm → MaxPool(1,8)
卷积层 2: Conv2d(8→16, kernel=(1,16)) → ELU → BatchNorm
卷积层 3: Conv2d(16→16, kernel=(1,16)) → ELU → BatchNorm
卷积层 4: Conv2d(16→32, kernel=(1,16)) → ELU → BatchNorm
卷积层 5: Conv2d(32→64, kernel=(1,8))  → ELU → BatchNorm → MaxPool(1,6)
卷积层 6: Conv2d(64→64, kernel=(1,8))  → ELU → BatchNorm
卷积层 7: Conv2d(64→128, kernel=(1,4)) → ELU → BatchNorm
卷积层 8: Conv2d(128→128, kernel=(1,4)) → ELU → BatchNorm → MaxPool(1,4)

Flatten → Linear(20224→64) → ELU → Dropout(0.5) → Linear(64→2)

输出: 2类分类 (0=噪声, 1=信号)
```

#### 3.1.2 关键配置

| 配置项 | 值 |
|--------|-----|
| 优化器 | Adam（Adaptive Moment Estimation，自适应矩估计） |
| 学习率 | 0.003 |
| 学习率调度 | CosineAnnealingLR（余弦退火学习率） |
| 损失函数 | CrossEntropyLoss（交叉熵损失） |
| Dropout | 0.5 |
| 激活函数 | ELU（Exponential Linear Unit，指数线性单元） |

---

### 3.2 ResNet（Residual Network，残差网络）模型

**文件**: `main_resnet.py`

用于BNS信号检测的ResNet架构，特点：
- 使用残差连接（Residual Connection）解决深层网络梯度消失问题
- 更适合处理BNS的高频信号特征
- 训练脚本: `train_resnet.py`
- 评估脚本: `evaluate_resnet.py`

---

## 四、模块引用关系

### 4.1 文件依赖图

```
                    ┌─────────────────┐
                    │ data_prep_bbh.py│
                    │  (BBH数据生成)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     main.py     │
                    │ (CNN模型+数据流) │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐  ┌───────▼───────┐
│   train.py    │   │   evaluate.py   │  │generate_submission.py│
│   (训练脚本)   │   │   (评估脚本)     │  │   (推理提交)   │
└───────────────┘   └─────────────────┘  └───────────────┘

                    ┌─────────────────┐
                    │ data_prep_bns.py│
                    │  (BNS数据生成)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  main_resnet.py │
                    │(ResNet模型+数据流)│
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐   ┌────────▼────────┐  ┌───────▼───────────┐
│train_resnet.py│   │evaluate_resnet.py│  │generate_submission.py│
└───────────────┘   └─────────────────┘  └───────────────────┘
```

### 4.2 调用链示例

```
main.py:DatasetGenerator.generate()
    └── data_prep_bbh.py:sim_data()
        ├── gen_psd()        # 为每个探测器生成PSD
        ├── gen_noise()      # 生成纯噪声样本
        ├── gen_par()        # 随机生成信号参数
        ├── gen_bbh()        # 生成BBH波形
        │   └── make_bbh()   # 应用探测器响应
        └── whiten_data()    # 数据白化处理
```

---

## 五、参数配置

### 5.1 数据生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fs` | 8192 Hz | 采样频率（Sampling Frequency） |
| `T_obs` | 1s | 观测时长（Observation Duration），实际使用 safe=2，即2秒 |
| `detectors` | [H1, L1] | 探测器（汉福德Hanford、利文斯顿Livingston） |
| `Nnoise` | 25 | 每个信号叠加25组噪声实现 |
| `mdist` | 'metric' | 质量分布模式 |
| `snr` | 20 | 目标信噪比（Signal-to-Noise Ratio） |

### 5.2 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 批次大小 |
| `lr` | 0.003 | 学习率（Learning Rate） |
| `total_epochs` | 100 | 总训练轮数 |
| `nsample_perepoch` | 100 | 每轮生成的样本数 |

---

## 六、模型评估指标

### 6.1 ROC（Receiver Operating Characteristic，受试者工作特征）曲线

- **横轴 (X轴)**: FPR（False Positive Rate，假阳性率）
  - FPR = FP / (FP + TN)
  
- **纵轴 (Y轴)**: TPR（True Positive Rate，真阳性率）
  - TPR = TP / (TP + FN)

### 6.2 AUC（Area Under the Curve，曲线下面积）指标

| AUC值 | 含义 |
|--------|------|
| 1.0 | 完美分类器 |
| 0.9-0.99 | 优秀 |
| 0.8-0.9 | 良好 |
| 0.7-0.8 | 一般 |
| **0.5** | **随机猜测** |
| < 0.5 | 比随机差（可能标签反了）|

### 6.3 SNR（Signal-to-Noise Ratio，信噪比）对性能的影响

| SNR | AUC | 说明 |
|-----|-----|------|
| 5 | 0.59 | 信号很弱，勉强比随机好 |
| 10 | 0.82 | 信号较弱，可接受 |
| 15 | 0.98 | 信号清晰，分类效果好 |
| 20 | 0.99 | 信号非常清晰，几乎完美 |

### 6.4 常见问题排查

如果 AUC < 0.5：

1. **检查模型加载** - 确认加载的是训练好的权重
2. **检查SNR设置** - 尝试提高SNR到20
3. **检查标签一致性** - 训练/测试时标签0/1含义是否一致
4. **检查数据分布** - 测试数据与训练数据分布是否一致

---

## 七、训练基础概念

### 7.1 Epoch

**Epoch** 指整个训练数据集被完整传递一次的过程。

#### Epoch、Batch、Iteration 的关系

$$
\text{总迭代次数} = \frac{\text{数据集大小}}{\text{Batch Size}} \times \text{Epoch 数}
$$

| 概念 | 英文全称 | 说明 |
|------|---------|------|
| **Batch Size** | - | 一次迭代使用的样本数量 |
| **Iteration** | - | 模型参数更新一次 |
| **Epoch** | - | 遍历完整数据集一次 |

**示例**：1000 张图片，Batch Size = 100
- 1 个 Epoch = 10 次 Iteration
- 模型看到 1000 张图片，更新 10 次参数

### 7.2 早停法（Early Stopping）

防止过拟合（Overfitting）的策略：
1. 划分训练集和验证集
2. 每个 Epoch 后计算验证集损失
3. 验证集损失不再下降时停止训练

### 7.3 欠拟合与过拟合

| 问题 | 英文 | 原因 | 解决 |
|------|------|------|------|
| 欠拟合 | Underfitting | Epoch 太少 | 增加训练轮数 |
| 过拟合 | Overfitting | Epoch 太多 | 早停、正则化（Regularization）、Dropout |

---

## 附录

### A. 核心文件清单

| 文件 | 功能 |
|------|------|
| `data_prep_bbh.py` | BBH数据生成 |
| `data_prep_bns.py` | BNS数据生成 |
| `main.py` | CNN模型定义 |
| `main_resnet.py` | ResNet模型定义 |
| `train.py` | CNN训练脚本 |
| `train_resnet.py` | ResNet训练脚本 |
| `evaluate.py` | CNN评估脚本 |
| `evaluate_resnet.py` | ResNet评估脚本 |
| `generate_submission.py` | 推理提交 |
| `utils.py` | 工具函数 |

### B. 依赖库

```
- lal / lalsimulation: LIGO算法库
- torch / torchvision: PyTorch深度学习框架
- numpy: 数值计算
- scipy: 信号处理
```

### C. 缩写词汇表

| 缩写 | 英文全称 | 中文翻译 |
|------|---------|---------|
| BBH | Binary Black Hole | 双黑洞 |
| BNS | Binary Neutron Star | 双中子星 |
| GW | Gravitational Wave | 引力波 |
| PSD | Power Spectral Density | 功率谱密度 |
| SNR | Signal-to-Noise Ratio | 信噪比 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| ResNet | Residual Network | 残差网络 |
| ROC | Receiver Operating Characteristic | 受试者工作特征 |
| AUC | Area Under the Curve | 曲线下面积 |
| FPR | False Positive Rate | 假阳性率 |
| TPR | True Positive Rate | 真阳性率 |
| FFT | Fast Fourier Transform | 快速傅里叶变换 |
| IFFT | Inverse Fast Fourier Transform | 逆快速傅里叶变换 |
| ELU | Exponential Linear Unit | 指数线性单元 |
| ReLU | Rectified Linear Unit | 线性整流单元 |
| Adam | Adaptive Moment Estimation | 自适应矩估计 |
| PN | Post-Newtonian | 后牛顿近似 |
| M☉ | Solar Mass | 太阳质量 |
