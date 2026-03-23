根据代码分析，以下是 `data_prep_bbh.py` 中**引力波信号+噪声数据生成方法**的详细说明，以及**引用关系**：

---

## 一、数据生成方法详解

### 1. 噪声生成 (`gen_noise` + `gen_psd`)
```python
# 生成探测器噪声的PSD（功率谱密度）
psd = gen_psd(fs, T_obs, op='AdvDesign', det='H1')
# 使用 LALSimulation 库中的 AdvDesign 灵敏度曲线

# 根据PSD生成时域高斯噪声
noise = gen_noise(fs, T_obs, psd)
# 原理：在频域生成复高斯随机数 → 乘以sqrt(PSD/4) → 逆FFT得到时域噪声
```

### 2. BBH（双黑洞）信号生成 (`gen_bbh`)
```python
# 使用 LALSimulation 的 IMRPhenomD 波形近似生成引力波信号
hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
    m1*MSUN_SI, m2*MSUN_SI,  # 质量
    0, 0, 0, 0, 0, 0,         # 自旋（这里设为0）
    1e6*PC_SI,                # 距离（1Mpc）
    iota, phi, 0, 0, 0,       # 倾角、相位
    1/fs, f_low, f_low,       # 采样率、起始频率
    approximant=IMRPhenomD    # 波形模型
)
```

### 3. 探测器响应 (`make_bbh`)
- **天线响应**：根据天空位置(ra, dec)和极化角(psi)计算Fp、Fc响应函数
- **时间延迟**：计算信号到达不同探测器相对于地心的时间延迟

### 4. 信号+噪声合成 (`sim_data`)
这是**主数据生成函数**，生成流程如下：

```
┌─────────────────────────────────────────────────────────┐
│  1. 噪声类样本 (label=0):                                │
│     - 纯噪声 → 白化 → 保存                               │
│                                                         │
│  2. 信号类样本 (label=1):                                │
│     - 随机生成BBH参数(m1, m2, 位置, 角度等)              │
│     - 生成波形(hp, hc) → 应用探测器响应                  │
│     - 归一化到目标SNR                                    │
│     - 每个信号叠加 Nnoise 组不同的噪声实现               │
│     - 白化处理 → 保存                                    │
└─────────────────────────────────────────────────────────┘
```

### 5. 质量分布 (`gen_masses`)
支持3种质量分布模式：
| 模式 | 说明 |
|------|------|
| `astro` | 天体物理对数分布 |
| `gh` | George & Huerta分布 (质量比q~[1,10]) |
| `metric` | 度规基础分布 |

---

## 二、引用关系

### 直接引用
| 文件 | 引用方式 | 用途 |
|------|---------|------|
| **`main.py`** | `from data_prep_bbh import *` | 在 `DatasetGenerator.generate()` 中调用 `sim_data()` 实时生成训练数据 |
| **`data_prep_bbh.py` 自身** | 内部调用 | `sim_data()` 调用 `gen_par()`, `gen_bbh()`, `gen_noise()`, `whiten_data()` 等 |

### 间接引用（通过 `main.py`）
| 文件 | 说明 |
|------|------|
| `evaluate.py` | 通过 `from main import DatasetGenerator` 生成测试数据评估模型 |
| `generate_submission.py` | 主要做推理，不直接生成训练数据 |
| `train.py` | 训练脚本（可能也使用了 `DatasetGenerator`）|

### 调用链示例
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

## 三、关键参数（默认值）

```python
fs = 8192          # 采样频率 (Hz)
T_obs = 1s         # 观测时长（实际使用 safe=2，即2秒）
detectors = [H1, L1]  # 探测器（汉福德、利文斯顿）
Nnoise = 25        # 每个信号叠加25组噪声实现
mdist = 'metric'   # 质量分布
snr = 20           # 目标信噪比
```

---

现在我来详细解释ROC曲线和AUC，以及你的结果：

---

## 1. ROC曲线是什么？

**ROC** = Receiver Operating Characteristic（受试者工作特征曲线）

- **横轴 (X轴)**: False Positive Rate (FPR) - 假阳性率
  - FPR = FP / (FP + TN)
  - "把负样本错误地预测为正样本"的比例
  
- **纵轴 (Y轴)**: True Positive Rate (TPR) - 真阳性率（也叫Recall/召回率）
  - TPR = TP / (TP + FN)
  - "把正样本正确地预测为正样本"的比例

---

## 2. AUC是什么？

**AUC** = Area Under the Curve（曲线下面积）

| AUC值 | 含义 |
|--------|------|
| 1.0 | 完美分类器 |
| 0.9-0.99 | 优秀 |
| 0.8-0.9 | 良好 |
| 0.7-0.8 | 一般 |
| **0.5** | **随机猜测**（和抛硬币一样）|
| < 0.5 | 比随机还差（可能标签反了）|

---

## 3. 为什么你的图是"直线"？

你的图不是严格的直线，而是一条**接近对角线的曲线**（AUC=0.4661）。

**这表示你的模型几乎是在随机猜测，甚至比随机猜测还差！**

可能的原因：
1. **模型没有训练好** - 权重是随机的，或者训练epoch太少
2. **标签反了** - 检查训练时的标签是否正确
3. **数据问题** - 训练数据和测试数据分布不一致
4. **SNR设置** - 你的信噪比可能太低，信号太弱

---

## 4. 和ipynb里的图对比

第二张图（ipynb里的）展示了**不同SNR（信噪比）**下的ROC曲线：

| SNR | AUC | 说明 |
|-----|-----|------|
| 5 | 0.59 | 信号很弱，勉强比随机好 |
| 10 | 0.82 | 信号较弱，但可以接受 |
| 15 | 0.98 | 信号清晰，分类效果好 |
| 20 | 0.99 | 信号非常清晰，几乎完美 |

**结论**：信噪比越高，信号越容易识别，AUC越高，ROC曲线越靠近左上角。

---

## 5. Luck / Random 是什么？

- **Random（随机）**：对角线，AUC=0.50
  - 表示"瞎猜"的水平
  - 如果你完全随机猜测，就会得到这条线
  
- **Luck（运气）**：和Random是一个意思
  - 在第二张图里标注为"Luck"

---

## 6. 你该怎么办？

你的AUC=0.4661 < 0.5，说明有问题。建议检查：

1. **检查模型是否正确加载** - 确认加载的是训练好的权重，不是随机初始化的
2. **检查SNR设置** - 尝试提高SNR（比如设为20），看AUC是否改善
3. **检查标签** - 确认训练时标签0/1的含义和测试时一致
4. **检查数据生成** - 确认测试数据生成代码正确

可以尝试运行：
```python
# 提高SNR看看效果
snr = 20  # 原来是多少？
```

如果SNR=20时AUC还是很低，那问题出在模型训练或加载上。