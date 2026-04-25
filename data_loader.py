"""
数据加载与预处理模块
- 读取 EuroSAT_RGB 数据集（10 类，每类若干 64x64 RGB 图片）
- 划分训练集 / 验证集 / 测试集
- 图像展平为向量，像素值归一化到 [0, 1]
"""

import os
import numpy as np
from PIL import Image


# 10 个类别（按文件夹名排序）
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 64
INPUT_DIM = IMG_SIZE * IMG_SIZE * 3  # 12288


def load_dataset(data_dir: str, max_per_class: int = None, seed: int = 42):
    """
    从 data_dir 下的各类别子文件夹加载所有图片。

    Returns
    -------
    X : ndarray, shape (N, 12288), float32, 值域 [0, 1]
    y : ndarray, shape (N,), int32, 类别索引
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"类别目录不存在: {cls_dir}")

        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".png", ".tif"))])
        if max_per_class is not None:
            idx = rng.choice(len(files), min(max_per_class, len(files)), replace=False)
            files = [files[i] for i in sorted(idx)]

        label = CLASS2IDX[cls]
        for fname in files:
            img_path = os.path.join(cls_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                arr = np.array(img, dtype=np.float32).reshape(-1) / 255.0  # (12288,)
                X_list.append(arr)
                y_list.append(label)
            except Exception as e:
                print(f"警告: 无法读取 {img_path}: {e}")

    X = np.stack(X_list, axis=0)   # (N, 12288)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def split_dataset(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    按比例划分训练集 / 验证集 / 测试集（分层抽样）。

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    train_idx, val_idx, test_idx = [], [], []

    for cls in range(NUM_CLASSES):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train:n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val:].tolist())

    def _shuffle(idx):
        idx = np.array(idx)
        rng.shuffle(idx)
        return idx

    ti = _shuffle(train_idx)
    vi = _shuffle(val_idx)
    tsi = _shuffle(test_idx)

    return (X[ti], y[ti]), (X[vi], y[vi]), (X[tsi], y[tsi])


def normalize(X_train, X_val, X_test):
    """
    用训练集的均值 / 标准差做标准化（z-score）。
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_test_n = (X_test - mean) / std
    return X_train_n, X_val_n, X_test_n, mean, std
