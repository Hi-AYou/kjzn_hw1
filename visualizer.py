"""
可视化模块
- 绘制训练/验证 Loss 曲线和验证 Accuracy 曲线
- 权重可视化（将第一层权重恢复为图像尺寸并显示）
- 类别相关权重可视化（按类别贡献度挑选第一层神经元）
- 错例分析（从测试集中挑出分类错误的样本并展示）
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import CLASSES, IMG_SIZE


# ──────────────────────────────────────────
# Loss & Accuracy 曲线
# ──────────────────────────────────────────

def plot_training_curves(history: dict, save_path: str = "training_curves.png"):
    """
    绘制：
    - 左图：训练集和验证集的 Loss 曲线
    - 右图：验证集的 Accuracy 曲线
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss 曲线
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    val_acc_pct = [v * 100 for v in history["val_acc"]]
    axes[1].plot(epochs, val_acc_pct, color="green", linewidth=1.5, label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Validation Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存至 {save_path}")


# ──────────────────────────────────────────
# 权重可视化（第一层）
# ──────────────────────────────────────────

def visualize_weights(W1: np.ndarray, save_path: str = "weights_visualization.png",
                      n_show: int = 32):
    """
    将第一层权重矩阵 W1 (input_dim, H1) 的每一列
    reshape 为 (64, 64, 3) 的图像并可视化。

    Parameters
    ----------
    W1      : (12288, H1)
    n_show  : 显示前 n_show 个神经元的权重图
    """
    H1 = W1.shape[1]
    n_show = min(n_show, H1)
    ncols = 8
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6))
    axes = axes.flatten()

    for i in range(n_show):
        w = W1[:, i].reshape(IMG_SIZE, IMG_SIZE, 3)
        # 归一化到 [0, 1] 以便显示
        w_min, w_max = w.min(), w.max()
        if w_max > w_min:
            w = (w - w_min) / (w_max - w_min)
        else:
            w = w - w_min
        axes[i].imshow(w)
        axes[i].axis("off")
        axes[i].set_title(f"#{i}", fontsize=7)

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("First Layer Weight Visualization (each neuron reshaped to 64×64×3)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"权重可视化图已保存至 {save_path}")


# ──────────────────────────────────────────
# 类别相关权重可视化
# ──────────────────────────────────────────

def visualize_class_weights(
    W1: np.ndarray,
    W2: np.ndarray,
    W3: np.ndarray,
    target_classes: list = None,
    n_neurons: int = 8,
    save_path: str = "class_weights_visualization.png",
):
    """
    通过 W3 和 W2 反向追溯，找出对每个目标类别贡献最大的 H1 神经元，
    将其权重 reshape 为 64×64×3 图像并可视化，分析各类别的颜色/纹理偏好。

    Parameters
    ----------
    W1             : (input_dim, H1)  — 第一层权重
    W2             : (H1, H2)         — 第二层权重
    W3             : (H2, num_classes)— 第三层权重
    target_classes : 要分析的类别名列表，None 时分析全部 10 类
    n_neurons      : 每个类别展示贡献最大的前 n 个 H1 神经元
    """
    if target_classes is None:
        target_classes = CLASSES

    # 计算每个 H1 神经元对每个类别的综合贡献分数：
    #   contribution[h1, c] = sum_h2 ( |W2[h1, h2]| * |W3[h2, c]| )
    # 即通过 H2 层传播的路径权重绝对值之积求和，衡量 H1 神经元
    # 经过两层线性变换后对类别 c 输出的潜在影响强度。
    contribution = np.abs(W2) @ np.abs(W3)   # (H1, num_classes)

    n_cls = len(target_classes)
    ncols = n_neurons
    nrows = n_cls

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.8, nrows * 1.8 + 0.6))

    for row, cls_name in enumerate(target_classes):
        cls_idx = CLASSES.index(cls_name)
        scores  = contribution[:, cls_idx]               # (H1,)
        top_idx = np.argsort(scores)[-n_neurons:][::-1]  # 贡献最大的前 n 个

        for col, h1_idx in enumerate(top_idx):
            w = W1[:, h1_idx].reshape(IMG_SIZE, IMG_SIZE, 3)
            w_min, w_max = w.min(), w.max()
            if w_max > w_min:
                w = (w - w_min) / (w_max - w_min)
            else:
                w = np.zeros_like(w)

            ax = axes[row, col]
            ax.imshow(w)
            ax.axis("off")

            # 分析该权重图的颜色通道均值，用于标注颜色倾向
            r_mean = w[:, :, 0].mean()
            g_mean = w[:, :, 1].mean()
            b_mean = w[:, :, 2].mean()
            dominant = ["R", "G", "B"][np.argmax([r_mean, g_mean, b_mean])]
            ax.set_title(f"H1#{h1_idx}\n{dominant}↑", fontsize=6)

        # 行标签（类别名）
        axes[row, 0].set_ylabel(cls_name, fontsize=8, rotation=0,
                                labelpad=60, va="center")

    plt.suptitle(
        "Top H1 Neurons per Class (ranked by contribution through W2·W3)\n"
        "Title shows neuron index and dominant RGB channel",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"类别相关权重可视化图已保存至 {save_path}")


# ──────────────────────────────────────────
# 错例分析
# ──────────────────────────────────────────

def error_analysis(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    n_show: int = 16,
    save_path: str = "error_analysis.png",
    mean=None, std=None,
):
    """
    从测试集中找出分类错误的样本，展示原始图像及其真实/预测标签。

    Parameters
    ----------
    X_test  : (N, 12288) — 标准化后的数据
    mean, std : 用于反标准化，若为 None 则直接 clip 到 [0,1]
    """
    error_idx = np.where(y_pred != y_test)[0]
    print(f"\n测试集中共有 {len(error_idx)} 个错误样本（共 {len(y_test)} 个）")

    n_show = min(n_show, len(error_idx))
    if n_show == 0:
        print("没有错误样本，跳过错例分析。")
        return

    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i, idx in enumerate(error_idx[:n_show]):
        x = X_test[idx].copy()
        if mean is not None and std is not None:
            x = x * std + mean           # 反标准化
        x = x.reshape(IMG_SIZE, IMG_SIZE, 3)
        x = np.clip(x, 0, 1)

        true_cls  = CLASSES[y_test[idx]]
        pred_cls  = CLASSES[y_pred[idx]]
        axes[i].imshow(x)
        axes[i].axis("off")
        axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls}", fontsize=8, color="red")

    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Error Analysis: Misclassified Samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"错例分析图已保存至 {save_path}")
