# HW1：从零构建三层 MLP — 卫星图像分类

> 纯 NumPy 实现三层多层感知机（MLP），完成 EuroSAT 遥感图像 10 分类任务。
> **不依赖 PyTorch / TensorFlow / JAX**，自主实现前向传播、反向传播与梯度计算。

---

## 项目地址

**GitHub Repo**：[https://github.com/Hi-AYou/kjzn_hw1](https://github.com/Hi-AYou/kjzn_hw1)

**模型权重（Google Drive）**：[点击下载 best_model.npz](https://drive.google.com/drive/folders/1OHgKIYI8lV6QxD7adL9WdBczMfKI1RDX?usp=sharing)

> 权重文件因体积较大（~25MB）未上传至 GitHub，请从上方 Google Drive 链接下载后放置于 `outputs/best_model.npz`，即可直接运行 `test.py`。

---

## 快速开始

### 环境依赖

```bash
Python >= 3.8
pip install numpy Pillow matplotlib
```

### 目录结构

```
kjzn_hw1/
├── data_loader.py          # 数据加载、分层划分、Z-score 标准化
├── model.py                # 三层 MLP：前向传播 + 手写反向传播
├── trainer.py              # SGD 优化器（带动量）+ 学习率衰减 + 训练循环
├── hp_search.py            # 网格搜索 / 随机搜索超参数模块
├── evaluator.py            # 测试集评估、准确率、混淆矩阵
├── visualizer.py           # 训练曲线、权重可视化、类别权重分析、错例分析
├── train.py                # 主训练脚本（一键运行完整流程）
├── test.py                 # 独立测试脚本（加载已保存权重进行评估）
├── 实验报告.md              # 完整实验报告
└── outputs/
    ├── training_curves.png             # 训练曲线
    ├── confusion_matrix.png            # 混淆矩阵热图
    ├── weights_visualization.png       # 第一层权重总览（32 个神经元）
    ├── class_weights_visualization.png # 各类别贡献最大 H1 神经元权重图
    ├── error_analysis.png              # 典型错误分类样本
    └── hp_search_results.json          # 超参数搜索结果
```

### 运行训练

```bash
# 确保 EuroSAT_RGB 数据集与本项目在同一目录下（即 ./EuroSAT_RGB）
python train.py
```

该脚本依次执行：
1. 加载并划分数据集（70 / 15 / 15 分层抽样）
2. 网格搜索超参数（16 组，每组 30 epoch）
3. 用最优超参数完整训练（80 epoch）
4. 评估测试集，输出准确率和混淆矩阵
5. 生成所有可视化图表（包含类别相关权重分析）

### 仅运行测试（已有权重）

```bash
# 先从 Google Drive 下载 best_model.npz 放到 outputs/
python test.py
```

---

## 模块说明

### `data_loader.py` — 数据加载与预处理

- `load_dataset(data_dir)` — 读取 10 类子文件夹，展平为 `(N, 12288)` float32 数组
- `split_dataset(X, y)` — 分层抽样，按 70/15/15 划分训练/验证/测试集
- `normalize(X_train, X_val, X_test)` — Z-score 标准化（仅用训练集统计量）

### `model.py` — 三层 MLP（核心）

网络架构：
```
Input (12288) → Linear → ReLU → Linear → ReLU → Linear → Softmax → Output (10)
                 ↑H1=512               ↑H2=128
```

- 支持 `relu` / `sigmoid` / `tanh` 激活函数切换
- He 初始化（针对 ReLU）
- 手写反向传播，交叉熵 + L2 正则化损失
- 参数保存/加载（`.npz`）

### `trainer.py` — 训练循环

- `SGDOptimizer` — SGD + 动量（momentum=0.9）
- `step_lr_decay` — Step Decay 学习率衰减
- `train()` — mini-batch 训练循环，自动保存验证集最优权重

### `hp_search.py` — 超参数搜索

- `grid_search()` — 枚举所有超参数组合（本次 16 组）
- `random_search()` — 从连续分布随机采样（适合搜索空间大时使用）

### `evaluator.py` — 测试评估

- `evaluate()` — 返回准确率、混淆矩阵、预测标签
- `plot_confusion_matrix()` — 绘制热力图并保存

### `visualizer.py` — 可视化

| 函数 | 作用 |
|------|------|
| `plot_training_curves(history)` | 训练/验证 Loss 曲线 + 验证 Accuracy 曲线 |
| `visualize_weights(W1, n_show=32)` | 第一层权重总览（前 32 个神经元 reshape 为图像） |
| `visualize_class_weights(W1, W2, W3, ...)` | 通过 W2·W3 贡献分数追溯各类别最相关 H1 神经元并可视化 |
| `error_analysis(X_test, y_test, y_pred, mean, std)` | 展示典型错误分类样本 |

---

## 实验结果汇总

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

### 最优超参数

```
lr = 0.005 | hidden1 = 512 | hidden2 = 128
weight_decay = 1e-4 | activation = relu
lr_decay_rate = 0.5 | lr_decay_step = 30 | epochs = 80
```

---

## 注意事项

1. **数据路径**：`train.py` / `test.py` 默认数据目录为 `./EuroSAT_RGB`（与脚本同级），如路径不同请修改 `DATA_DIR`。
2. **模型权重**：`best_model.npz` 未包含在仓库中，请从 Google Drive 下载后放置于 `outputs/`。
3. **复现性**：所有随机操作均设置固定种子（`seed=42`），结果可完全复现。
4. **内存需求**：完整数据集约需 1.2GB 内存（27,000 × 12,288 × 4 bytes）。
