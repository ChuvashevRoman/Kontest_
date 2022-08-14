import numpy as np
import pandas as pd
import os
from scipy.signal import convolve2d
import time

def mse_loss(y_true, y_pred):
    # y_true и y_pred - массивы numpy одинаковой длины
    return ((y_true - y_pred) ** 2).mean()

def sigmoid(x):
    # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x * k))
    k = 1
    return 1 / (1 + np.exp(-x * k))

def sigmoid_der(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

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

# Данные с матрицами свёрток
x_conv_1 = []
x_conv_2 = []

for i in range(len(x_steps_1)):
    x_conv_1.append(convolve2d(x_steps_1[i], np.ones((3, 3)), mode='same', boundary='wrap'))
    x_conv_2.append(convolve2d(x_steps_1[i], np.ones((5, 5)), mode='same', boundary='wrap'))
x_conv_1 = np.array(x_conv_1)
x_conv_2 = np.array(x_conv_2)

# Матрицы альфа, бета, гамма из исходных данных
alpha = np.tile(np.floor(np.arange(20) / (20 / 3)) + 1, (20, 1))
beta =  alpha.T + 1
alpha, beta = np.minimum(alpha, beta), np.maximum(alpha, beta)
gamma = (alpha + beta) / 2

alpha = np.array([alpha for i in range(len(x_steps_1))])
beta = np.array([beta for i in range(len(x_steps_1))])
gamma = np.array([gamma for i in range(len(x_steps_1))])

# Входные данные
input_data = [x_steps_1, x_conv_1, x_conv_2, alpha, beta, gamma]

# Размеры нейронной сети
len_input = len(input_data) # количество входных данных
len_h1 = 12 # Количество нейронов в первом слое
len_h2 = 12 # Количество нейронов во втором слое
len_output = 1

# Веса и коэфф-ты смещения
w_1 = np.random.normal(size=(len_input, len_h1))
b_1 = np.random.normal(size=(1, len_h1))
w_2 = np.random.normal(size=(len_h1, len_h2))
b_2 = np.random.normal(size=(1, len_h2))
w_3 = np.random.normal(size=(len_h2, len_output))
b_3 = np.random.normal(size=(1, len_output))

# Шаг спуска и количество эпох
lr = 0.05
steps = 10

for step in range(steps):
    start = time.time()
    for n in range(len(x_steps_1)):
        input_i = []
        for item in input_data:
            input_i.append(item[n])
        y_pred = np.random.normal(size=(20,20))
        for i in range(len(input_i[0])):
            for j in range(len(input_i[0][i])):
                # Прямой проход
                x = np.array([[input_i[k][i][j] for k in range(len(input_i))]])
                t_1 = x @ w_1 + b_1
                h_1 = sigmoid(t_1)
                t_2 = h_1 @ w_2 + b_2
                h_2 = sigmoid(t_2)
                t_3 = h_2 @ w_3 + b_3
                o_1 = sigmoid(t_3)
                y_pred[i][j] = o_1

                # Обратный ход
                sigma_3 = (y_steps_1[0][i][j] - o_1) * sigmoid_der(t_3)
                d_w3 = h_2.T @ sigma_3
                d_b3 = np.sum(sigma_3, axis=0, keepdims=True)
                sigma_2 = sigma_3 @ w_3.T * sigmoid_der(t_2)
                d_w2 = h_1.T @ sigma_2
                d_b2 = np.sum(sigma_2, axis=0, keepdims=True)
                sigma_1 = sigma_2 @ w_2.T * sigmoid_der(t_1)
                d_w1 = x.T @ sigma_1
                d_b1 = np.sum(sigma_1, axis=0, keepdims=True)

                # Обновление весов
                w_1 = w_1 + lr * d_w1
                b_1 = b_1 + lr * d_b1
                w_2 = w_2 + lr * d_w2
                b_2 = b_2 + lr * d_b2
                w_3 = w_3 + lr * d_w3
                b_3 = b_3 + lr * d_b3

        mse = mse_loss(y_pred, y_steps_1[0])
        print("==============================================")
        print(f"Ошибка на шаге {n}: {mse}")
    if step < steps - 1:
        print("*********************************")
        print(f"УШЛА ЭПОХА {step + 1}")
        print(f"ПРИШЛА ЭПОХА {step + 2}")
        print(f"Время выполнения эпохи: {time.time() - start}")
        print("*********************************")
print("*********************************")
print("Обучение закончено")

def feedforwarf(input_i, w, b):
    w_1, w_2, w_3 = w[0], w[1], w[2]
    b_1, b_2, b_3 = b[0], b[1], b[2]
    y_pred = np.random.normal(size=(20, 20))
    for i in range(len(input_i[0])):
        for j in range(len(input_i[0][i])):
            # Прямой проход
            x = np.array([[input_i[k][i][j] for k in range(len(input_i))]])
            t_1 = x @ w_1 + b_1
            h_1 = sigmoid(t_1)
            t_2 = h_1 @ w_2 + b_2
            h_2 = sigmoid(t_2)
            t_3 = h_2 @ w_3 + b_3
            o_1 = sigmoid(t_3)
            y_pred[i][j] = o_1
    return y_pred

w = [w_1, w_2, w_3]
b = [b_1, b_2, b_3]

y_predict = np.random.normal(size=(len(x_steps_1),20,20))
for i in range(len(x_steps_1)):
    input_i = [input_data[0][i], input_data[1][i], input_data[2][i],
               input_data[3][i], input_data[4][i], input_data[5][i]]
    y_predict[i] = feedforwarf(input_i, w, b)


mse = mse_loss(y_predict, np.array(y_steps_1))



