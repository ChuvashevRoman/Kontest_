import numpy as np
import scipy
from scipy.signal import convolve2d
from scipy.optimize import minimize

class NerualNetwork_2:
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
      return ((y_true - y_pred) ** 2).mean()

    def feedforward(self, w, b):
        X_conv_1 = convolve2d(self.X_matrix, np.ones((3, 3)), mode='same', boundary='wrap')
        X_conv_2 = convolve2d(self.X_matrix, np.ones((5, 5)), mode='same', boundary='wrap')
        h1 = self.sigmoid(self.X_matrix * w[0] + X_conv_1 * w[1] + X_conv_2 * w[2] + b[0])
        h2 = self.sigmoid(self.X_matrix * w[3] + X_conv_1 * w[4] + X_conv_2 * w[5] + b[1])
        h3 = self.sigmoid(self.X_matrix * w[6] + X_conv_1 * w[7] + X_conv_2 * w[8] + b[2])
        o1 = self.sigmoid(h1 * w[9] + h2 * w[10] + h3 * w[11] + b[3])
        return o1

    def func(self, w, b):
        return self.mse_loss(self.feedforward(w, b), self.Y_matrix)

    def gradient(self, f, w, b):
        h = 1e-4 #h - разница независимой переменной при выводе, ее можно задать как очень маленькую константу
        grad_w = np.zeros_like(w) #
        for i in range(w.size):
            tmp_val = w[i]
            w[i] = tmp_val + h
            fxh1 = f(w, b) # Вычислить f (x + h)
            w[i] = tmp_val - h
            fxh2 = f(w, b) # Вычислить f (x-h)
            grad_w[i] = (fxh1 - fxh2)/(2*h)
            w[i] = tmp_val #reduction
        grad_b = np.zeros_like(b) #
        for i in range(b.size):
            tmp_val = b[i]
            b[i] = tmp_val + h
            fxh1 = f(w, b) # Вычислить f (x + h)
            b[i] = tmp_val - h
            fxh2 = f(w, b) # Вычислить f (x-h)
            grad_b[i] = (fxh1 - fxh2)/(2*h)
            b[i] = tmp_val #reduction
        return grad_w, grad_b

    def lern_NN(self, lr = 0.01, step_num=10000):
        w_array = np.array([np.random.normal() for i in range(12)])
        b_array = np.array([np.random.normal() for i in range(4)])
        for i in range(step_num):
            grad_w, grad_b = self.gradient(self.func, w_array, b_array)
            w_array -= lr * grad_w
            b_array -= lr * grad_b
            print(i)
        print(w_array)
        return w_array, b_array