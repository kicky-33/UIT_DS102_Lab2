import numpy as np
from tqdm import tqdm


class SoftmaxRegression:
    def __init__(self, epoch: int, lr: float, n_classes: int = 10):
        self.epoch = epoch
        self.lr = lr
        self.n_classes = n_classes
        self.w = None
        self.losses = []

    def _softmax(self, z):
        # Trừ đi max(z) để tránh lỗi số mũ quá lớn gây tràn số (NaN)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _one_hot(self, y):
        # Chuyển nhãn số (0-9) thành vector 10 chiều
        m = y.shape[0]
        one_hot = np.zeros((m, self.n_classes))
        one_hot[np.arange(m), y.astype(int)] = 1
        return one_hot

    def loss_fn(self, y_one_hot, a):
        # Hàm mất mát Categorical Cross-Entropy
        m = y_one_hot.shape[0]
        epsilon = 1e-15
        return -np.sum(y_one_hot * np.log(a + epsilon)) / m

    def fit(self, X, y):
        m, d = X.shape
        # Khởi tạo ma trận trọng số (d x 10)
        self.w = np.zeros((d, self.n_classes))
        y_one_hot = self._one_hot(y)

        for e in tqdm(range(self.epoch), desc="Training Softmax"):
            # Forward pass
            z = X @ self.w
            a = self._softmax(z)

            # Backward pass: Tính Gradient
            gradient = (X.T @ (a - y_one_hot)) / m

            # Cập nhật trọng số
            self.w -= self.lr * gradient

            # Lưu Loss
            loss = self.loss_fn(y_one_hot, a)
            self.losses.append(loss)

    def predict(self, X):
        # Dự đoán nhãn có xác suất cao nhất
        probs = self._softmax(X @ self.w)
        return np.argmax(probs, axis=1)
