# Lab 2: Logistic & Softmax Regression - MNIST Classification
## Thông tin sinh viên
Họ và tên: La Gia Hân

MSSV: 24520448

Lớp: KHDL2024 - Đại học Công nghệ Thông tin (UIT)

## Mô tả dự án
Bài Lab này tập trung vào việc xây dựng các mô hình phân loại chữ số viết tay trên bộ dữ liệu MNIST bằng cả phương pháp thủ công (Numpy) và sử dụng thư viện (Scikit-Learn).

### Các nội dung chính:
Assignment 1: Xây dựng mô hình Logistic Regression bằng Numpy để phân loại nhị phân (chữ số 0 và 1).

Assignment 2: Xây dựng mô hình Softmax Regression bằng Numpy để phân loại đa lớp (toàn bộ 10 chữ số từ 0-9).

Assignment 3: Triển khai lại bài toán bằng thư viện Scikit-Learn để đối chiếu và đánh giá hiệu năng.

## Kết quả thực nghiệm

Dựa trên quá trình huấn luyện với các Hyperparameters khác nhau, dưới đây là kết quả tối ưu đạt được:

- Logistic Regression (0 & 1): F1-score đạt xấp xỉ 0.9991 với $lr=2.0$ và $epoch=50$.

- Softmax Regression (10 lớp): Accuracy đạt khoảng 0.91 với $lr=0.5$ và $epoch=200$.

- Scikit-Learn: Cho kết quả ổn định và hội tụ nhanh hơn nhờ các thuật toán solver tối ưu (lbfgs).
