"""
独立测试脚本
加载最优模型权重，对测试集进行评估。
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, split_dataset, normalize, INPUT_DIM, NUM_CLASSES
from model import MLP
from evaluator import evaluate, plot_confusion_matrix
from visualizer import visualize_weights, visualize_class_weights, error_analysis

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EuroSAT_RGB")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.npz")
NORM_MEAN_PATH  = os.path.join(OUTPUT_DIR, "norm_mean.npy")
NORM_STD_PATH   = os.path.join(OUTPUT_DIR, "norm_std.npy")


def main():
    # 加载数据（与训练时相同的分割）
    print("加载数据集...")
    X, y = load_dataset(DATA_DIR)
    _, _, (X_test, y_test) = split_dataset(X, y)

    # 使用训练时保存的均值/标准差
    mean = np.load(NORM_MEAN_PATH)
    std  = np.load(NORM_STD_PATH)
    X_test = (X_test - mean) / std

    # 加载模型
    print(f"加载模型权重: {BEST_MODEL_PATH}")
    model = MLP.load(BEST_MODEL_PATH, input_dim=INPUT_DIM, num_classes=NUM_CLASSES)

    # 评估
    acc, conf_mat, y_pred, probs = evaluate(model, X_test, y_test, verbose=True)

    # 保存混淆矩阵
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_confusion_matrix(conf_mat, save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    # 错例分析
    error_analysis(
        X_test, y_test, y_pred,
        n_show=16,
        save_path=os.path.join(OUTPUT_DIR, "error_analysis.png"),
        mean=mean, std=std,
    )

    # 权重可视化
    visualize_weights(
        model.W1,
        save_path=os.path.join(OUTPUT_DIR, "weights_visualization.png"),
        n_show=32,
    )
    visualize_class_weights(
        model.W1, model.W2, model.W3,
        target_classes=["Forest", "SeaLake", "River", "Highway",
                        "AnnualCrop", "HerbaceousVegetation", "PermanentCrop",
                        "Industrial", "Residential", "Pasture"],
        n_neurons=8,
        save_path=os.path.join(OUTPUT_DIR, "class_weights_visualization.png"),
    )

    print(f"\n测试完成，最终准确率: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
