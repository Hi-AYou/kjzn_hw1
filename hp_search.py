"""
超参数搜索模块
- 支持网格搜索 (Grid Search) 或随机搜索 (Random Search)
- 记录每组超参数下的验证集准确率
- 返回最优超参数组合
"""

import itertools
import numpy as np
import os
from model import MLP
from trainer import train
from data_loader import INPUT_DIM, NUM_CLASSES


def grid_search(
    X_train, y_train, X_val, y_val,
    param_grid: dict,
    epochs: int = 40,
    batch_size: int = 256,
    save_dir: str = "hp_search",
    verbose: bool = True,
):
    """
    网格搜索。

    param_grid 示例：
    {
        "lr":           [1e-2, 5e-3],
        "hidden1":      [256, 512],
        "hidden2":      [128, 256],
        "weight_decay": [1e-4, 1e-3],
        "activation":   ["relu"],
    }

    Returns
    -------
    results : list[dict]  — 每组超参数及对应 val_acc
    best    : dict        — 最优超参数
    """
    os.makedirs(save_dir, exist_ok=True)

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combos = list(itertools.product(*values))

    results = []
    best_val_acc = -1.0
    best_params = None

    print(f"网格搜索：共 {len(combos)} 组超参数")
    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        if verbose:
            print(f"\n[{idx+1}/{len(combos)}] 超参数: {params}")

        model = MLP(
            input_dim=INPUT_DIM,
            hidden1=params.get("hidden1", 256),
            hidden2=params.get("hidden2", 128),
            num_classes=NUM_CLASSES,
            activation=params.get("activation", "relu"),
            weight_decay=params.get("weight_decay", 1e-4),
            seed=42,
        )

        save_path = os.path.join(save_dir, f"model_combo{idx+1}.npz")
        history = train(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr_init=params.get("lr", 1e-2),
            lr_decay_rate=params.get("lr_decay_rate", 0.5),
            lr_decay_step=params.get("lr_decay_step", 20),
            save_path=save_path,
            verbose=False,
        )

        val_acc = max(history["val_acc"])
        results.append({**params, "val_acc": val_acc, "save_path": save_path,
                        "history": history})

        if verbose:
            print(f"  最优验证集准确率: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {**params, "save_path": save_path}

    print(f"\n网格搜索完成！最优超参数: {best_params}, val_acc={best_val_acc*100:.2f}%")
    return results, best_params


def random_search(
    X_train, y_train, X_val, y_val,
    param_distributions: dict,
    n_iter: int = 20,
    epochs: int = 40,
    batch_size: int = 256,
    save_dir: str = "hp_search",
    verbose: bool = True,
    seed: int = 0,
):
    """
    随机搜索。

    param_distributions 示例：
    {
        "lr":           ("log_uniform", 1e-3, 1e-1),
        "hidden1":      ("choice", [128, 256, 512]),
        "hidden2":      ("choice", [64, 128, 256]),
        "weight_decay": ("log_uniform", 1e-5, 1e-2),
        "activation":   ("choice", ["relu", "tanh"]),
    }

    Returns
    -------
    results : list[dict]
    best    : dict
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    def sample_param(spec):
        kind = spec[0]
        if kind == "choice":
            return rng.choice(spec[1])
        elif kind == "log_uniform":
            lo, hi = np.log(spec[1]), np.log(spec[2])
            return float(np.exp(rng.uniform(lo, hi)))
        elif kind == "uniform":
            return float(rng.uniform(spec[1], spec[2]))
        else:
            raise ValueError(f"未知的分布类型: {kind}")

    results = []
    best_val_acc = -1.0
    best_params = None

    print(f"随机搜索：共 {n_iter} 次")
    for idx in range(n_iter):
        params = {k: sample_param(v) for k, v in param_distributions.items()}
        # 确保整数类型
        for k in ("hidden1", "hidden2"):
            if k in params:
                params[k] = int(params[k])

        if verbose:
            print(f"\n[{idx+1}/{n_iter}] 超参数: {params}")

        model = MLP(
            input_dim=INPUT_DIM,
            hidden1=params.get("hidden1", 256),
            hidden2=params.get("hidden2", 128),
            num_classes=NUM_CLASSES,
            activation=params.get("activation", "relu"),
            weight_decay=params.get("weight_decay", 1e-4),
            seed=42,
        )

        save_path = os.path.join(save_dir, f"model_rand{idx+1}.npz")
        history = train(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr_init=params.get("lr", 1e-2),
            lr_decay_rate=0.5,
            lr_decay_step=20,
            save_path=save_path,
            verbose=False,
        )

        val_acc = max(history["val_acc"])
        results.append({**params, "val_acc": val_acc, "save_path": save_path,
                        "history": history})

        if verbose:
            print(f"  最优验证集准确率: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {**params, "save_path": save_path}

    print(f"\n随机搜索完成！最优超参数: {best_params}, val_acc={best_val_acc*100:.2f}%")
    return results, best_params
