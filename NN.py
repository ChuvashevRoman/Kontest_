import numpy as np
import scipy
from scipy.signal import convolve2d

class NerualNetwork:
    def __init__(self, X_matrix, Y_matrix_true):
        # Веса
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()
        # Пороги
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()
        # Входная и выходная матрица
        self.X_matrix = X_matrix
        self.Y_matrix = Y_matrix

    def sigmoid(x):
      # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
      return 1 / (1 + np.exp(-x))

    def relu(x):
        return (np.maximum(0, x)

    def relu_lim(x):
        return np.minimum(np.maximum(0, x), 1)

    def deriv_sigmoid(x):
      # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
      fx = sigmoid(x)
      return fx * (1 - fx)

    def mse_loss(y_true, y_pred):
      # y_true и y_pred - массивы numpy одинаковой длины
      return ((y_true - y_pred) ** 2).mean()

    def convolve_1(self, X):
        nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap')

    def convolve_2(self, X):
        nbrs_count = convolve2d(X, np.ones((5, 5)), mode='same', boundary='wrap')

    def feedforward(self, x):
        X_conv_1 = self.convolve_1(self.X_matrix)
        X_conv_2 = self.convolve_2(self.X_matrix)
        h1 = self.sigmoid(self.X_matrix * self.w1 + X_conv_1 * self.w2 + X_conv_2 * self.w3 + self.b1)
        h2 = self.sigmoid(self.X_matrix * self.w4 + X_conv_1 * self.w5 + X_conv_2 * self.w6 + self.b2)
        h3 = self.sigmoid(self.X_matrix * self.w7 + X_conv_1 * self.w8 + X_conv_2 * self.w9 + self.b3)
        o1 = self.sigmoid(self.X_matrix * self.w10 + X_conv_1 * self.w11 + X_conv_2 * self.w12  + self.b4)
        return o1