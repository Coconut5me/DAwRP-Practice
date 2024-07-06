import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu giả
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.1, 3.9, 6.0, 7.9, 10.1])

# Khởi tạo tham số
w = 0.0
b = 0.0

# Tốc độ học
alpha = 0.01

# Số lần lặp
epochs = 100

# Lưu trữ lịch sử MSE để vẽ đồ thị
mse_history = []

# Gradient Descent
for epoch in range(epochs):
    y_pred = w * x + b
    error = y - y_pred
    mse = np.mean(error ** 2)
    mse_history.append(mse)

    grad_w = (-2 / len(x)) * np.dot(error, x)
    grad_b = (-2 / len(x)) * np.sum(error)

    w -= alpha * grad_w
    b -= alpha * grad_b

# Vẽ đồ thị MSE qua mỗi epoch
plt.plot(mse_history)
plt.title('MSE qua các lần lặp')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()
