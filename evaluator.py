"""
测试评估模块
- 输出测试集准确率
- 打印混淆矩阵
- 保存混淆矩阵热图
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from data_loader import CLASSES, NUM_CLASSES
from model import MLP


def evaluate(model: MLP, X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True):
    """
    Returns
    -------
    acc        : float
    conf_mat   : ndarray (NUM_CLASSES, NUM_CLASSES)
    y_pred     : ndarray (N,)
    probs      : ndarray (N, NUM_CLASSES)
    """
    probs = model.forward(X_test, training=False)
    y_pred = np.argmax(probs, axis=1)
    acc = float((y_pred == y_test).mean())

    # 混淆矩阵
    conf_mat = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for true, pred in zip(y_test, y_pred):
        conf_mat[true, pred] += 1

    if verbose:
        print(f"\n测试集准确率: {acc*100:.2f}%\n")
        print("混淆矩阵（行=真实类别，列=预测类别）：")
        header = " " * 24 + "  ".join(f"{c[:6]:>6}" for c in CLASSES)
        print(header)
        for i, row in enumerate(conf_mat):
            row_str = "  ".join(f"{v:>6}" for v in row)
            print(f"  {CLASSES[i]:<22} {row_str}")

    return acc, conf_mat, y_pred, probs


def plot_confusion_matrix(conf_mat: np.ndarray, save_path: str = "confusion_matrix.png"):
    """绘制并保存混淆矩阵热图"""
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(conf_mat, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)

    thresh = conf_mat.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(conf_mat[i, j]),
                    ha="center", va="center",
                    color="white" if conf_mat[i, j] > thresh else "black",
                    fontsize=7)

    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title("Confusion Matrix on Test Set", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"混淆矩阵图已保存至 {save_path}")
