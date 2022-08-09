import pandas as pd
import os

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

# Матрица входных данных
X_matrix = x_steps_1[0]

# Матрица выходных данных
Y_matrix = y_steps_1[0]

