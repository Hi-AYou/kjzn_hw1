"""
主训练脚本
1. 加载数据并划分数据集
2. 超参数网格搜索（快速版）
3. 用最优超参数完整训练模型
4. 评估测试集（准确率 + 混淆矩阵）
5. 可视化（训练曲线 / 权重 / 错例分析）
"""

import os
import sys
import numpy as np

# 将项目目录加入 PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, split_dataset, normalize, INPUT_DIM, NUM_CLASSES
from model import MLP
from trainer import train
from hp_search import grid_search
from evaluator import evaluate, plot_confusion_matrix
from visualizer import plot_training_curves, visualize_weights, error_analysis

# ────────────────────────────────────────────────────────
# 路径配置（根据实际情况修改 DATA_DIR）
# ────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "EuroSAT_RGB")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.npz")
HP_SEARCH_DIR   = os.path.join(OUTPUT_DIR, "hp_search")


# ────────────────────────────────────────────────────────
# Step 1: 加载数据
# ────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: 加载数据集...")
X, y = load_dataset(DATA_DIR)
print(f"  数据集大小: {X.shape}, 类别数: {len(set(y.tolist()))}")

(X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)
print(f"  训练集: {X_train.shape[0]}  验证集: {X_val.shape[0]}  测试集: {X_test.shape[0]}")

X_train, X_val, X_test, mean, std = normalize(X_train, X_val, X_test)
np.save(os.path.join(OUTPUT_DIR, "norm_mean.npy"), mean)
np.save(os.path.join(OUTPUT_DIR, "norm_std.npy"),  std)
print("  数据标准化完成。")


# ────────────────────────────────────────────────────────
# Step 2: 超参数搜索（网格搜索，较快版本）
# ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: 超参数网格搜索...")

PARAM_GRID = {
    "lr":           [1e-2, 5e-3],
    "hidden1":      [512, 256],
    "hidden2":      [256, 128],
    "weight_decay": [1e-4, 1e-3],
    "activation":   ["relu"],
    "lr_decay_rate": [0.5],
    "lr_decay_step": [20],
}

hp_results, best_params = grid_search(
    X_train, y_train, X_val, y_val,
    param_grid=PARAM_GRID,
    epochs=30,             # 搜索阶段轮次较少，节省时间
    batch_size=256,
    save_dir=HP_SEARCH_DIR,
    verbose=True,
)

# 保存搜索结果（不含 history 和 save_path 等非 JSON 友好字段）
import json
hp_summary = [{k: v for k, v in r.items() if k not in ("history", "save_path")}
              for r in hp_results]
with open(os.path.join(OUTPUT_DIR, "hp_search_results.json"), "w", encoding="utf-8") as f:
    json.dump(hp_summary, f, indent=2, ensure_ascii=False)
print(f"超参数搜索结果已保存至 {OUTPUT_DIR}/hp_search_results.json")


# ────────────────────────────────────────────────────────
# Step 3: 用最优超参数完整训练
# ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 3: 使用最优超参数进行完整训练...\n  {best_params}")

best_model = MLP(
    input_dim=INPUT_DIM,
    hidden1=best_params["hidden1"],
    hidden2=best_params["hidden2"],
    num_classes=NUM_CLASSES,
    activation=best_params.get("activation", "relu"),
    weight_decay=best_params.get("weight_decay", 1e-4),
    seed=42,
)

history = train(
    best_model,
    X_train, y_train,
    X_val,   y_val,
    epochs=80,
    batch_size=256,
    lr_init=best_params["lr"],
    lr_decay_rate=best_params.get("lr_decay_rate", 0.5),
    lr_decay_step=best_params.get("lr_decay_step", 20),
    momentum=0.9,
    save_path=BEST_MODEL_PATH,
    verbose=True,
)

plot_training_curves(history, save_path=os.path.join(OUTPUT_DIR, "training_curves.png"))


# ────────────────────────────────────────────────────────
# Step 4: 加载最优权重并在测试集上评估
# ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: 在测试集上评估...")

final_model = MLP.load(BEST_MODEL_PATH, input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
acc, conf_mat, y_pred, probs = evaluate(final_model, X_test, y_test, verbose=True)

plot_confusion_matrix(conf_mat, save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


# ────────────────────────────────────────────────────────
# Step 5: 可视化分析
# ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: 生成可视化图表...")

# 5a. 第一层权重可视化
visualize_weights(
    final_model.W1,
    save_path=os.path.join(OUTPUT_DIR, "weights_visualization.png"),
    n_show=32,
)

# 5b. 错例分析
error_analysis(
    X_test, y_test, y_pred,
    n_show=16,
    save_path=os.path.join(OUTPUT_DIR, "error_analysis.png"),
    mean=mean, std=std,
)

print("\n" + "=" * 60)
print("全部流程完成！输出文件保存在:", OUTPUT_DIR)
print(f"  - best_model.npz          : 最优模型权重")
print(f"  - training_curves.png     : 训练 Loss 和 Val Accuracy 曲线")
print(f"  - confusion_matrix.png    : 测试集混淆矩阵")
print(f"  - weights_visualization.png: 第一层权重可视化")
print(f"  - error_analysis.png      : 错例分析图")
print(f"  - hp_search_results.json  : 超参数搜索结果")
print(f"\n最终测试集准确率: {acc*100:.2f}%")
