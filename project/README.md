# HW1：从零构建三层 MLP — 卫星图像分类

> 纯 NumPy 实现三层多层感知机（MLP），完成 EuroSAT 遥感图像 10 分类任务。
> **不依赖 PyTorch / TensorFlow / JAX**，自主实现前向传播、反向传播与自动微分。

---

## 快速开始

### 环境依赖

```bash
Python >= 3.8
pip install numpy Pillow matplotlib
```

### 目录结构

```
hw1/
├── EuroSAT_RGB/                # 数据集（10 个类别子文件夹）
│   ├── AnnualCrop/
│   ├── Forest/
│   └── ...（共 10 个）
└── project/                    # 代码目录
    ├── data_loader.py
    ├── model.py
    ├── trainer.py
    ├── hp_search.py
    ├── evaluator.py
    ├── visualizer.py
    ├── train.py                # 主训练脚本
    ├── test.py                 # 独立测试脚本
    └── outputs/                # 训练输出（自动创建）
```

### 运行训练

```bash
cd project/
python train.py
```

该脚本会依次执行：
1. 加载并划分数据集（70/15/15）
2. 网格搜索超参数（16 组，每组 30 epoch）
3. 用最优超参数完整训练（120 epoch）
4. 评估测试集，输出准确率和混淆矩阵
5. 生成所有可视化图表

### 运行测试（已有权重）

```bash
cd project/
python test.py
```

---

## 模块说明

### `data_loader.py` — 数据加载与预处理

**核心功能**：
- `load_dataset(data_dir)` — 遍历 10 个类别子文件夹，读取所有 jpg 图像，展平为 `(N, 12288)` 的 float32 数组
- `split_dataset(X, y)` — 分层抽样，按 70/15/15 比例划分训练/验证/测试集
- `normalize(X_train, X_val, X_test)` — 使用训练集的均值和标准差对所有数据做 Z-score 标准化

**关键参数**：
```python
IMG_SIZE  = 64          # 图像边长
INPUT_DIM = 12288       # 64 × 64 × 3
NUM_CLASSES = 10
CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation",
           "Highway", "Industrial", "Pasture", "PermanentCrop",
           "Residential", "River", "SeaLake"]
```

---

### `model.py` — 三层 MLP（核心）

**网络架构**：
```
Input (12288) → Linear → ReLU → Linear → ReLU → Linear → Softmax → Output (10)
                 ↑H1=512           ↑H2=128
```

**支持的激活函数**（通过 `activation` 参数切换）：
- `"relu"` — ReLU（默认，训练效果最好）
- `"sigmoid"` — Sigmoid
- `"tanh"` — Tanh

**主要方法**：

| 方法 | 说明 |
|------|------|
| `__init__(input_dim, hidden1, hidden2, num_classes, activation, weight_decay)` | 初始化网络参数（He 初始化） |
| `forward(X, training=True)` | 前向传播，返回 softmax 概率；training=True 时缓存中间变量用于反向传播 |
| `backward(probs, y)` | 反向传播，返回各参数的梯度字典 |
| `loss(probs, y)` | 计算交叉熵 + L2 正则化总损失 |
| `predict(X)` | 返回预测类别（argmax） |
| `save(path)` | 保存权重到 `.npz` 文件 |
| `MLP.load(path, input_dim, num_classes)` | 从文件加载模型 |

**反向传播关键推导**（以下是代码中的数学逻辑）：

```
# Softmax + CrossEntropy 联合梯度（N 为 batch size）
dz3 = (probs - one_hot(y)) / N

# 第三层
dW3 = a2.T @ dz3 + λ·W3
db3 = dz3.sum(axis=0)

# 反传过激活函数
dz2 = (dz3 @ W3.T) * act'(z2)

# 第二层
dW2 = a1.T @ dz2 + λ·W2
...（以此类推）
```

---

### `trainer.py` — 训练循环

**`SGDOptimizer`** — SGD + 动量：
```python
v = momentum · v - lr · grad
param += v
```

**`step_lr_decay`** — Step 学习率衰减：
```python
lr = lr_init × decay_rate ^ (epoch // step)
```

**`train()` 函数参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 60 | 训练轮数 |
| `batch_size` | 256 | mini-batch 大小 |
| `lr_init` | 1e-2 | 初始学习率 |
| `lr_decay_rate` | 0.5 | 学习率衰减率 |
| `lr_decay_step` | 20 | 每隔多少 epoch 衰减一次 |
| `momentum` | 0.9 | SGD 动量系数 |
| `save_path` | `"best_model.npz"` | 最优模型保存路径 |

每个 epoch 结束后在验证集上计算准确率，**自动保存验证集最优权重**（Early Stopping 的简化版本）。

---

### `hp_search.py` — 超参数搜索

**网格搜索** `grid_search()`：枚举所有超参数组合，对每组训练固定轮数，记录最优验证集准确率。

```python
PARAM_GRID = {
    "lr":           [1e-2, 5e-3],
    "hidden1":      [512, 256],
    "hidden2":      [256, 128],
    "weight_decay": [1e-4, 1e-3],
    "activation":   ["relu"],
}
results, best_params = grid_search(X_train, y_train, X_val, y_val, param_grid=PARAM_GRID, epochs=30)
```

**随机搜索** `random_search()`：从连续分布中随机采样超参数（适合搜索空间较大时使用）。

```python
PARAM_DIST = {
    "lr":           ("log_uniform", 1e-3, 1e-1),
    "hidden1":      ("choice", [128, 256, 512]),
    "weight_decay": ("log_uniform", 1e-5, 1e-2),
}
results, best_params = random_search(..., param_distributions=PARAM_DIST, n_iter=20)
```

---

### `evaluator.py` — 测试评估

**`evaluate(model, X_test, y_test)`** — 返回：
- `acc` — 整体准确率（float）
- `conf_mat` — 混淆矩阵（10×10 ndarray）
- `y_pred` — 预测标签数组
- `probs` — 预测概率数组

**`plot_confusion_matrix(conf_mat)`** — 绘制并保存带数值标注的热图。

---

### `visualizer.py` — 可视化

| 函数 | 作用 |
|------|------|
| `plot_training_curves(history)` | 绘制训练/验证 Loss 曲线 + 验证 Accuracy 曲线 |
| `visualize_weights(W1, n_show=32)` | 将第一层权重 reshape 为 64×64×3 图像并展示 |
| `error_analysis(X_test, y_test, y_pred, mean, std)` | 展示分类错误的样本及其真实/预测标签 |

---

## 实验结果汇总

### 最终性能

| 指标 | 值 |
|------|-----|
| **测试集准确率** | **64.72%** |
| 最优验证集准确率 | 65.28% |
| 训练集 Loss（末期） | ~0.12 |
| 验证集 Loss（最优） | ~1.18 |

### 各类别准确率

| 类别 | 准确率 |
|------|--------|
| Forest | 89.3% ⭐ |
| SeaLake | 83.3% ⭐ |
| Industrial | 75.5% |
| Pasture | 70.7% |
| AnnualCrop | 65.8% |
| River | 60.3% |
| HerbaceousVegetation | 60.2% |
| Residential | 55.6% |
| Highway | 40.8% ⚠️ |
| PermanentCrop | 40.8% ⚠️ |

### 主要混淆对

- **Highway ↔ River**：线性结构相似（56 次误分）
- **PermanentCrop → AnnualCrop / HerbaceousVegetation**：农作物类别高度相似（150 次误分）
- **Industrial ↔ Residential**：建筑纹理相近（43 次误分）
- **SeaLake → Forest**：深色区域视觉相似（49 次误分）

---

## 超参数搜索结果

最优超参数组合（16 组中排名第一）：

```
lr = 0.005
hidden1 = 512
hidden2 = 128
weight_decay = 1e-4
activation = relu
lr_decay_rate = 0.5
lr_decay_step = 30
```

完整搜索结果见 `outputs/hp_search_results.json`。

---

## 输出文件说明

训练完成后，`outputs/` 目录下生成以下文件：

| 文件 | 说明 |
|------|------|
| `best_model.npz` | 验证集最优模型权重（~25MB） |
| `norm_mean.npy` | 训练集像素均值（用于推理时标准化） |
| `norm_std.npy` | 训练集像素标准差 |
| `training_curves.png` | 训练/验证 Loss 曲线 + 验证 Accuracy 曲线 |
| `confusion_matrix.png` | 测试集混淆矩阵热图 |
| `weights_visualization.png` | 第一层权重可视化（32 个神经元） |
| `error_analysis.png` | 16 张典型错误分类样本 |
| `hp_search_results.json` | 超参数搜索全部结果 |
| `hp_search/` | 各超参数组合对应的模型权重 |

---

## 注意事项

1. **数据路径**：`train.py` 和 `test.py` 默认数据目录为脚本所在目录的上一级 `../EuroSAT_RGB`，即 `hw1/EuroSAT_RGB`。如路径不同请修改 `train.py` 中的 `DATA_DIR` 变量。
2. **首次运行**：`train.py` 会自动执行超参数搜索（约需 10-20 分钟，取决于机器性能），如只需用已找到的最优参数训练，可将 `train.py` 中的超参数搜索部分注释掉。
3. **复现性**：所有随机操作均设置了固定随机种子（`seed=42`），结果可完全复现。
4. **内存需求**：完整数据集约需 1.2GB 内存（27,000 × 12,288 × 4 bytes），请确保可用内存充足。
