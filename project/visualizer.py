"""
可视化模块
- 绘制训练/验证 Loss 曲线和验证 Accuracy 曲线
- 权重可视化（将第一层权重恢复为图像尺寸并显示）
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
