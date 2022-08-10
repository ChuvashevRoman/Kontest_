import numpy as np
import scipy
from scipy.signal import convolve2d
from scipy.optimize import minimize

class NerualNetwork:
    def __init__(self, X_matrix, Y_matrix):
        # Веса
        self.w = [np.random.normal() for i in range(12)]
        # Пороги
        self.b = [np.random.normal() for i in range(12)]
        # Входная и выходная матрица
        self.X_matrix = X_matrix
        self.Y_matrix = Y_matrix

    def sigmoid(self, x):
      # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
      return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_lim(self, x):
        return np.minimum(np.maximum(0, x), 1)

    # def deriv_sigmoid(x):
    #   # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
    #   fx = self.sigmoid(x)
    #   return fx * (1 - fx)

    def mse_loss(self, y_true, y_pred):
      # y_true и y_pred - массивы numpy одинаковой длины
      return float(((y_true - y_pred) ** 2).mean())

    def feedforward(self):
        X_conv_1 = convolve2d(self.X_matrix, np.ones((3, 3)), mode='same', boundary='wrap')
        X_conv_2 = convolve2d(self.X_matrix, np.ones((5, 5)), mode='same', boundary='wrap')
        h1 = self.sigmoid(self.X_matrix * self.w[0] + X_conv_1 * self.w[1] + X_conv_2 * self.w[2] + self.b[0])
        h2 = self.sigmoid(self.X_matrix * self.w[3] + X_conv_1 * self.w[4] + X_conv_2 * self.w[5] + self.b[1])
        h3 = self.sigmoid(self.X_matrix * self.w[6] + X_conv_1 * self.w[7] + X_conv_2 * self.w[8] + self.b[2])
        o1 = self.sigmoid(h1 * self.w[9] + h2 * self.w[10] + h3 * self.w[11] + self.b[3])
        return o1

    def func(self, X):
        return self.mse_loss(self.feedforward(), self.Y_matrix)

    def lern_NN(self):
        print(self.w + self.b)
        self.func(5)
        opt = minimize(self.func, self.w + self.b, method="Nelder-Mead")
        self.func(5)
        return minimize(self.func, self.w + self.b, method="Nelder-Mead")