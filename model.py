"""
模型定义模块
从零实现三层 MLP（全连接神经网络）。
- 支持自定义隐藏层大小
- 支持 ReLU / Sigmoid / Tanh 激活函数切换
- 手动实现前向传播与反向传播（自动微分）
- 不依赖 PyTorch / TensorFlow / JAX 等框架
"""

import numpy as np


# ──────────────────────────────────────────
# 激活函数
# ──────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(z.dtype)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    "relu":    (relu,    relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh":    (tanh,    tanh_grad),
}


# ──────────────────────────────────────────
# Softmax + Cross-Entropy Loss
# ──────────────────────────────────────────

def softmax(z):
    """数值稳定版 softmax，z: (N, C)"""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    """
    Parameters
    ----------
    probs : (N, C) — softmax 输出
    y     : (N,)   — 整数类别标签

    Returns
    -------
    loss : scalar
    """
    N = len(y)
    log_p = -np.log(np.clip(probs[np.arange(N), y], 1e-15, 1.0))
    return log_p.mean()


# ──────────────────────────────────────────
# 三层 MLP
# ──────────────────────────────────────────

class MLP:
    """
    三层 MLP: Input → Hidden1 → Hidden2 → Output

    架构
    ----
    z1 = X  @ W1 + b1          (N, H1)
    a1 = act(z1)
    z2 = a1 @ W2 + b2          (N, H2)
    a2 = act(z2)
    z3 = a2 @ W3 + b3          (N, C)
    probs = softmax(z3)

    Parameters
    ----------
    input_dim  : int   — 输入维度（12288）
    hidden1    : int   — 第一隐藏层大小
    hidden2    : int   — 第二隐藏层大小
    num_classes: int   — 类别数（10）
    activation : str   — 激活函数名称，支持 relu / sigmoid / tanh
    weight_decay: float — L2 正则化系数
    seed       : int   — 随机种子
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int,
        hidden2: int,
        num_classes: int,
        activation: str = "relu",
        weight_decay: float = 1e-4,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        if activation not in ACTIVATIONS:
            raise ValueError(f"不支持的激活函数: {activation}，可选: {list(ACTIVATIONS)}")
        self.activation_name = activation
        self._act, self._act_grad = ACTIVATIONS[activation]

        # 参数初始化（He / Xavier）
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden1)
        scale3 = np.sqrt(2.0 / hidden2)

        self.W1 = rng.standard_normal((input_dim,  hidden1)).astype(np.float32) * scale1
        self.b1 = np.zeros((hidden1,), dtype=np.float32)
        self.W2 = rng.standard_normal((hidden1, hidden2)).astype(np.float32) * scale2
        self.b2 = np.zeros((hidden2,), dtype=np.float32)
        self.W3 = rng.standard_normal((hidden2, num_classes)).astype(np.float32) * scale3
        self.b3 = np.zeros((num_classes,), dtype=np.float32)

        # 缓存前向传播的中间变量（用于反向传播）
        self._cache = {}

    # ── 前向传播 ──

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        X : (N, input_dim)

        Returns
        -------
        probs : (N, num_classes) — softmax 输出概率
        """
        z1 = X @ self.W1 + self.b1           # (N, H1)
        a1 = self._act(z1)                    # (N, H1)
        z2 = a1 @ self.W2 + self.b2          # (N, H2)
        a2 = self._act(z2)                    # (N, H2)
        z3 = a2 @ self.W3 + self.b3          # (N, C)
        probs = softmax(z3)                   # (N, C)

        if training:
            self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3}

        return probs

    # ── 反向传播 ──

    def backward(self, probs: np.ndarray, y: np.ndarray):
        """
        计算参数梯度（含 L2 正则化项）。

        Parameters
        ----------
        probs : (N, C) — 前向传播输出的 softmax 概率
        y     : (N,)  — 真实类别标签

        Returns
        -------
        grads : dict，键为 W1/b1/W2/b2/W3/b3
        """
        N = len(y)
        X, z1, a1, z2, a2, z3 = (
            self._cache["X"],
            self._cache["z1"],
            self._cache["a1"],
            self._cache["z2"],
            self._cache["a2"],
            self._cache["z3"],
        )

        # ∂L/∂z3 (softmax + cross-entropy 联合导数)
        dz3 = probs.copy()
        dz3[np.arange(N), y] -= 1
        dz3 /= N                              # (N, C)

        # 第三层参数梯度
        dW3 = a2.T @ dz3 + self.weight_decay * self.W3   # (H2, C)
        db3 = dz3.sum(axis=0)                             # (C,)

        # 反传到 a2
        da2 = dz3 @ self.W3.T                             # (N, H2)

        # 过激活函数
        dz2 = da2 * self._act_grad(z2)                    # (N, H2)

        # 第二层参数梯度
        dW2 = a1.T @ dz2 + self.weight_decay * self.W2   # (H1, H2)
        db2 = dz2.sum(axis=0)                             # (H2,)

        # 反传到 a1
        da1 = dz2 @ self.W2.T                             # (N, H1)

        # 过激活函数
        dz1 = da1 * self._act_grad(z1)                    # (N, H1)

        # 第一层参数梯度
        dW1 = X.T @ dz1 + self.weight_decay * self.W1    # (D, H1)
        db1 = dz1.sum(axis=0)                             # (H1,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    # ── Loss（含 L2 正则项）──

    def loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        ce = cross_entropy_loss(probs, y)
        l2 = 0.5 * self.weight_decay * (
            np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2) + np.sum(self.W3 ** 2)
        )
        return ce + l2

    # ── 预测 ──

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X, training=False)
        return np.argmax(probs, axis=1)

    # ── 参数保存 / 加载 ──

    def save(self, path: str):
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            activation=self.activation_name,
            weight_decay=self.weight_decay,
        )

    @classmethod
    def load(cls, path: str, input_dim: int, num_classes: int):
        data = np.load(path, allow_pickle=True)
        model = cls(
            input_dim=input_dim,
            hidden1=int(data["hidden1"]),
            hidden2=int(data["hidden2"]),
            num_classes=num_classes,
            activation=str(data["activation"]),
            weight_decay=float(data["weight_decay"]),
        )
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.W3 = data["W3"]
        model.b3 = data["b3"]
        return model
