import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
# Giả sử file class của bạn tên là logistic_model.py
from logistic_regression import LogisticRegression

train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
train_data = (train_images, train_labels)

test_images = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")
test_data = (test_images, test_labels)


# LOAD VÀ LỌC DỮ LIỆU

def filter_data(data, classes):
    images, labels = data
    # Tạo mask để lọc các nhãn nằm trong danh sách classes (0 và 1) [cite: 829-830]
    mask = np.isin(labels, classes)
    new_images = images[mask]
    new_labels = labels[mask]
    return new_images, new_labels


def preprocess_mnist(images):
    # Chuyển ảnh (28, 28) thành vector (784,)
    X = images.reshape(images.shape[0], -1)

    # Scaling: Chia cho 255 để đưa giá trị pixel về [0, 1]
    X = X.astype(np.float64) / 255.0

    # Add Bias: Thêm cột số 1 vào đầu ma trận X
    X = np.c_[np.ones((X.shape[0], 1)), X]
    return X


# Lọc lấy số 0 và 1 cho cả tập train và test
X_train_raw, y_train = filter_data((train_images, train_labels), [0, 1])
X_test_raw, y_test = filter_data((test_images, test_labels), [0, 1])

# Tiền xử lý
X_train = preprocess_mnist(X_train_raw)
X_test = preprocess_mnist(X_test_raw)

# HUẤN LUYỆN VÀ ĐÁNH GIÁ

model = LogisticRegression(epoch=50, lr=2.0)
model.fit(X_train, y_train)
# Dự đoán trên tập Test
y_hat_test = model.predict(X_test)

precision = precision_score(y_test, y_hat_test)
recall = recall_score(y_test, y_hat_test)
f1 = f1_score(y_test, y_hat_test)

print(f"KẾT QUẢ ASSIGNMENT 1")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(model.losses, label='Training Loss', color='tab:red', linewidth=2)
plt.title("Sự thay đổi của Hàm mất mát (Loss Function) qua từng Epoch")
plt.xlabel("Số lượng Epoch")
plt.ylabel("Giá trị Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()

# Chạy lần 1 với epoch=100, lr=0.1 [Figure1]: Pre=0.9982   Recall: 1.0000  F1: 0.9991
# Chạy lần 2 với epoch=100, lr=0.01 [Figure2]: Pre=0.9974   Recall: 0.9991  F1: 0.9982
# Chạy lần 3 với epoch=50, lr=1.5 [Figure3]: Pre=0.9991   Recall: 1.0000  F1: 0.9996
# Chạy lần 4 với epoch=150, lr=0.01 [Figure4]: Pre=0.9974   Recall: 1.0000  F1: 0.9987
# Chạy lần 5 với epoch=50, lr=2.0 [Figure5]: Pre=0.9991   Recall: 1.0000  F1: 0.9996

# Lần 5 đạt được F1-score tối đa (0.9991) chỉ trong vòng 50 epoch nhờ tốc độ học lớn (lr=2.0).
# -> tốn ít thời gian và tài nguyên tính toán nhất mà vẫn đạt kết quả tối ưu
