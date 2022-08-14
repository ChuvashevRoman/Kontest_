import numpy as np
import pandas as pd
import os
from scipy.signal import convolve2d
from numba import njit
import time

# Общий датафрейм с тренировочными данными
df = pd.read_csv(os.getcwd() + r'\data\train2022.csv')

# Датафрейм с данными в 1 шаг
steps_1 = df[df['steps'] == 1].reset_index(drop=True)

# Наименования входных и выходных данных
x_coll = []
y_coll = []
for item in df.columns:
    if "x_" in item:
        x_coll.append(item)
    if "y_" in item:
        y_coll.append(item)

# Массивы с бинарными матрицами
x_steps_1 = []
y_steps_1 = []

for item in steps_1[x_coll].to_numpy():
    x_steps_1.append(item.reshape(20, 20))
for item in steps_1[y_coll].to_numpy():
    y_steps_1.append(item.reshape(20, 20))

x_steps_1 = np.array(x_steps_1)
y_steps_1 = np.array(y_steps_1)

# Матрица входных данных
X_matrix = x_steps_1[0]

# Матрица выходных данных
Y_matrix = y_steps_1[0]

def mse_loss(y_true, y_pred):
    # y_true и y_pred - массивы numpy одинаковой длины
    return ((y_true - y_pred) ** 2).mean()

def sigmoid(x):
    # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x * k))
    k = 1
    return 1 / (1 + np.exp(-x * k))

def feed_dorward(input_data, w, b):
    # Прогон матрицы по нейронной сети
    # w - массив весов 6*9 + 9*9 + 9 = 153 шт
    # b - массив смещений 9 + 9 + 1 = 19 шт
    # X_conv_1 - свёртка матрицы 3х3
    h_1 = [] # Нейроны первого слоя
    h_2 = [] # Нейроны второго слоя

    # Расчёт нейронов первого слоя
    for i in range(9):
        h_i = b[0][i]
        for j in range(len(input_data)):
            h_i += input_data[j] * w[0][j + i * 6]
            h_i = sigmoid(h_i)
        h_1.append(h_i)

    # Второй слой
    for i in range(9):
        h_i = b[1][i]
        for j in range(len(h_1)):
            h_i += h_1[j] * w[1][j + i * 9]
            h_i = sigmoid(h_i)
        h_2.append(h_i)

    # Выходной нейрон
    o = b[2][0]
    for i in range(len(h_2)):
        o += h_2[i] * w[2][i]
    o = sigmoid(o)
    return o

def main_function(input_data, y, w, b):
    # Расчёт mse на всем датасете
    y_pred = feed_dorward(input_data, w, b)
    return mse_loss(y_pred, y)

def gradient(input_data, y, w, b):
    # Вычисления градиента от функции main_function
    h = 1e-2 #h - разница независимой переменной при выводе, ее можно задать как очень маленькую константу
    grad_w = []
    for i in range(len(w)):
        grad_w.append(np.zeros_like(w[i]))
        for j in range(w[i].size):
            tmp_val = w[i][j]
            w[i][j] = tmp_val + h
            f_w_1 = main_function(input_data, y, w, b) # Вычислить f (x + h)
            w[i][j] = tmp_val - h
            f_w_2 = main_function(input_data, y, w, b) # Вычислить f (x-h)
            grad_w[i][j] = (f_w_1 - f_w_2)/(2*h)
            w[i][j] = tmp_val #reduction
    grad_b = []
    for i in range(len(b)):
        grad_b.append(np.zeros_like(b[i]))
        for j in range(b[i].size):
            tmp_val = b[i][j]
            b[i][j] = tmp_val + h
            f_b_1 = main_function(input_data, y, w, b) # Вычислить f (x + h)
            b[i][j] = tmp_val - h
            f_b_2 = main_function(input_data, y, w, b) # Вычислить f (x-h)
            grad_b[i][j] = (f_b_1 - f_b_2)/(2*h)
            b[i][j] = tmp_val #reduction
    return grad_w, grad_b

def lern_nn(input_data, y_dataset, lr=0.01, steps=100):
    w = [np.zeros((9 * 6)),
         np.zeros((9 * 9)),
         np.zeros(9)]
    b = [np.zeros(9),
         np.zeros(9),
         np.zeros(1)]
    print(f"MSE на начало обучения: {main_function(input_data, y_dataset, w, b)}")
    for i in range(steps):
        start = time.time()
        for j in range(len(input_data[0])):
            grad_w, grad_b = gradient([input_data[0][j], input_data[1][j],
                                       input_data[2][j], input_data[3][j],
                                       input_data[4][j], input_data[5][j]],
                                      y_dataset[j], w, b)
            for i in range(len(w)):
                w[i] -= lr * grad_w[i]
            for i in range(len(b)):
                b[i] -= lr * grad_b[i]
            print(j)
            print(f"MSE: {main_function(input_data, y_dataset, w, b)}")
            if (i / len(input_data[0]) * 100) % 10 == 0:
                print(f"{i / len(input_data[0]) * 100}% Итерации")
        print(f"MSE на итерации {i}: {main_function(input_data, y_dataset, w, b)}")
        print(f"Затраченное время на итерацию: {time.time() - start}")
    return w, b


x = x_steps_1
y = y_steps_1

x_conv_1 = []
x_conv_2 = []

for i in range(len(x)):
    x_conv_1.append(convolve2d(x[i], np.ones((3, 3)), mode='same', boundary='wrap'))
    x_conv_2.append(convolve2d(x[i], np.ones((5, 5)), mode='same', boundary='wrap'))
x_conv_1 = np.array(x_conv_1)
x_conv_2 = np.array(x_conv_2)

alpha = np.tile(np.floor(np.arange(20) / (20 / 3)) + 1, (20, 1))
beta =  alpha.T + 1
alpha, beta = np.minimum(alpha, beta), np.maximum(alpha, beta)
gamma = (alpha + beta) / 2

alpha = np.array([alpha for i in range(len(x))])
beta = np.array([beta for i in range(len(x))])
gamma = np.array([gamma for i in range(len(x))])

input_data = [x, x_conv_1, x_conv_2, alpha, beta, gamma]

w, b = lern_nn(input_data, y, lr=1, steps=10)

# w = [np.random.normal(size=9 * 6),
#      np.random.normal(size=9 * 9),
#      np.random.normal(size=9)]
# b = [np.random.normal(size=9),
#      np.random.normal(size=9),
#      np.random.normal(size=1)]
#
# input = []
# for item in input_data:
#     input.append(item[0])
#
# y_pred = feed_dorward(input_data, w, b)