import numpy as np
import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load và Tiền xử lý dữ liệu


def load_and_preprocess():
    train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
    train_labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
    test_images = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")

    X_train = train_images.reshape(train_images.shape[0], -1) / 255.0
    X_test = test_images.reshape(test_images.shape[0], -1) / 255.0

    return X_train, train_labels, X_test, test_labels


X_train, y_train, X_test, y_test = load_and_preprocess()

# LOGISTIC REGRESSION

mask_train = (y_train == 0) | (y_train == 1)
mask_test = (y_test == 0) | (y_test == 1)

X_train_bin, y_train_bin = X_train[mask_train], y_train[mask_train]
X_test_bin, y_test_bin = X_test[mask_test], y_test[mask_test]

# Huấn luyện bằng Scikit-Learn
log_reg = LogisticRegression(max_iter=100)
log_reg.fit(X_train_bin, y_train_bin)

# Dự đoán và Đánh giá
y_pred_bin = log_reg.predict(X_test_bin)
print("LOGISTIC REGRESSION:")
print(classification_report(y_test_bin, y_pred_bin))


# SOFTMAX REGRESSION

softmax_reg = LogisticRegression(solver="lbfgs", C=10)
softmax_reg.fit(X_train, y_train)

# Dự đoán và Đánh giá
y_pred_sm = softmax_reg.predict(X_test)
print("\nSOFTMAX REGRESSION")
print(classification_report(y_test, y_pred_sm))

# Độ chính xác: Kết quả từ Sklearn cao hơn code Numpy tự viết
# vì Sklearn sử dụng các thuật toán tối ưu hóa (solvers) mạnh hơn thay vì chỉ dùng Gradient Descent cơ bản.
