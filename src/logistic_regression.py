import numpy as np
from tqdm import tqdm


class LogisticRegression:
    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.lr = lr
        self.w = None
        self.losses = []

    def _sigmoid(self, z):  # Đưa đầu ra về (0, 1)
        return 1 / (1 + np.exp(-z))

    def loss_fn(self, y: np.ndarray, y_hat: np.ndarray) -> float:  # Hàm mất mát Binary Cross-Entropy
        epsilon = 1e-15  # Tránh log(0)
        l = y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon)
        return -np.mean(l)

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, d = X.shape
        # Khởi tạo trọng số bằng 0
        self.w = np.zeros((d, 1), dtype=np.float64)
        y = y.reshape(-1, 1)  # Đảm bảo y có dạng (N, 1)

        for e in tqdm(range(self.epoch), desc="Training"):
            # Forward pass: Tính xác suất dự đoán p-hat
            y_hat = self.predict_proba(X)

            # Backward pass: Tính Vector Gradient
            # Công thức: (1/N) * X^T * (y_hat - y)
            gradient = (X.T @ (y_hat - y)) / N

            # Cập nhật trọng số: w = w - lr * gradient
            self.w -= self.lr * gradient

            # Lưu lại lịch sử mất mát
            loss = self.loss_fn(y, y_hat)
            self.losses.append(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # Trả về xác suất (0 đến 1)
        z = X @ self.w
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:  # Trả về nhãn lớp 0 hoặc 1
        y_proba = self.predict_proba(X)
        return (y_proba >= 0.5).astype(int)
