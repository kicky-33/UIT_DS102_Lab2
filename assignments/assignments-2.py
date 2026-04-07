import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from softmax_regression import SoftmaxRegression

# 1. Load dữ liệu
train_images = idx2numpy.convert_from_file("train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("train-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("t10k-labels.idx1-ubyte")

# 2. Tiền xử lý (Không lọc nhãn)


def preprocess(images):
    X = images.reshape(images.shape[0], -1)
    X = X.astype(np.float64) / 255.0  # Chuẩn hóa
    X = np.c_[np.ones((X.shape[0], 1)), X]  # Thêm Bias
    return X


X_train = preprocess(train_images)
X_test = preprocess(test_images)

model = SoftmaxRegression(epoch=200, lr=0.5)
model.fit(X_train, train_labels)

y_pred = model.predict(X_test)
print(classification_report(test_labels, y_pred))

plt.plot(model.losses)
plt.title("Softmax Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Chạy lần 1 với epoch=200, lr=0.5 [sm_Figure_1]: accuracy = 0.91
# Chạy lần 2 với epoch=200, lr=0.1 [sm_Figure_2]: accuracy = 0.89
# Chạy lần 3 với epoch=200, lr=0.7 [sm_Figure_3]: accuracy = 0.91
# Chạy lần 4 với epoch=250, lr=0.5 [sm_Figure_4]: accuracy = 0.91
# Chạy lần 5 với epoch=250, lr=0.3 [sm_Figure_5]: accuracy = 0.91

# Lần chạy 1 tối ưu về cả độ chính xác và tốc độ
