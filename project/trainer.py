"""
训练循环模块
- SGD 优化器
- 学习率衰减（Step Decay）
- 支持 mini-batch
- 根据验证集准确率自动保存最优权重
"""

import numpy as np
import os
from model import MLP


class SGDOptimizer:
    """SGD 优化器（支持动量）"""

    def __init__(self, model: MLP, lr: float = 1e-2, momentum: float = 0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        # 动量缓存
        self._v = {k: np.zeros_like(getattr(model, k))
                   for k in ("W1", "b1", "W2", "b2", "W3", "b3")}

    def step(self, grads: dict):
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            self._v[k] = self.momentum * self._v[k] - self.lr * grads[k]
            param = getattr(self.model, k)
            param += self._v[k]


def step_lr_decay(lr_init: float, epoch: int, decay_rate: float = 0.5, step: int = 20):
    """每隔 step 个 epoch 将学习率乘以 decay_rate"""
    return lr_init * (decay_rate ** (epoch // step))


def train(
    model: MLP,
    X_train, y_train,
    X_val,   y_val,
    *,
    epochs: int = 60,
    batch_size: int = 256,
    lr_init: float = 1e-2,
    lr_decay_rate: float = 0.5,
    lr_decay_step: int = 20,
    momentum: float = 0.9,
    save_path: str = "best_model.npz",
    verbose: bool = True,
    seed: int = 42,
):
    """
    训练三层 MLP，返回训练历史。

    Returns
    -------
    history : dict，包含 train_loss / val_loss / val_acc 列表
    """
    rng = np.random.default_rng(seed)
    opt = SGDOptimizer(model, lr=lr_init, momentum=momentum)
    N = len(y_train)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        # 更新学习率
        opt.lr = step_lr_decay(lr_init, epoch - 1, lr_decay_rate, lr_decay_step)

        # Shuffle
        perm = rng.permutation(N)
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        epoch_losses = []
        for start in range(0, N, batch_size):
            Xb = X_shuf[start: start + batch_size]
            yb = y_shuf[start: start + batch_size]

            probs = model.forward(Xb, training=True)
            loss = model.loss(probs, yb)
            grads = model.backward(probs, yb)
            opt.step(grads)
            epoch_losses.append(loss)

        train_loss = float(np.mean(epoch_losses))

        # 验证
        val_probs = model.forward(X_val, training=False)
        val_loss = model.loss(val_probs, y_val)
        val_preds = np.argmax(val_probs, axis=1)
        val_acc = float((val_preds == y_val).mean())

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(val_acc)

        # 保存最优权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(save_path)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d}/{epochs} | lr={opt.lr:.6f} | "
                f"train_loss={train_loss:.4f} | val_loss={float(val_loss):.4f} | "
                f"val_acc={val_acc*100:.2f}%"
            )

    if verbose:
        print(f"\n训练完成！最优验证集准确率: {best_val_acc*100:.2f}%，权重已保存至 {save_path}")

    return history
