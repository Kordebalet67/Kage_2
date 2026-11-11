import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('kaggleton_2_data.csv')
# print(df)
df.columns = ['x','y']

df.plot(x='x', y='y', kind='scatter', marker='x', color='green', s=100)
df['x_norm'] = (df['x'] - df['x'].mean()) / (df['x'].max() - df['x'].min())
m = len(df)

theta_0 = 1751
theta_1 = 1255
theta_2 = -4887
theta_3 = -8267

df['prediction'] = (theta_0 + 
                   theta_1 * np.sin(df['x_norm']) + 
                   theta_2 * np.sin(df['x_norm']**2) +
                   theta_3 * np.sin(df['x_norm']**3))


plt.plot(df['x'], df['prediction'], color='blue', linestyle='-', linewidth=3)




"""theta_0 = 1750.0
theta_1 = 1230.0
theta_2 = -4850.0
theta_3 = -8310.0
theta_0 = 1751
theta_1 = 1255
theta_2 = -4887
theta_3 = -8267
theta_4 = 0
theta_5 = 0
theta_6 = 0"""

n_iterations = 10_000
alpha = 1.5

loss_values = []
intermediate_models = []

for i in range(n_iterations):
    predictions = (theta_0 + 
                   theta_1 * np.sin(df['x_norm']) + 
                   theta_2 * np.sin(df['x_norm']**2) +
                   theta_3 * np.sin(df['x_norm']**3))
    
    loss = 1 / (2 * m) * ((predictions - df['y']) ** 2).sum()
    loss_values.append(loss)
    
    if i % 100 == 0:
        print(f'Iteration: {i}, loss: {loss}')
    
    gradient_theta_0 = (1 / m) * (predictions - df['y']).sum()
    gradient_theta_1 = (1 / m) * np.sum((predictions - df['y']) * df['x_norm'])
    gradient_theta_2 = (1 / m) * np.sum((predictions - df['y']) * (df['x_norm']**2))
    gradient_theta_3 = (1 / m) * np.sum((predictions - df['y']) * (df['x_norm']**3))
    
    theta_0 = theta_0 - alpha * gradient_theta_0
    theta_1 = theta_1 - alpha * gradient_theta_1
    theta_2 = theta_2 - alpha * gradient_theta_2
    theta_3 = theta_3 - alpha * gradient_theta_3
    
    intermediate_models.append((theta_0, theta_1, theta_2, theta_3))

print((theta_0, theta_1, theta_2, theta_3))
plt.figure(figsize=(10, 6))
# plt.plot(range(1, n_iterations+1), loss_values)
plt.plot(range(20, n_iterations+1), loss_values[19:])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()



plt.show()























"""
ВАРИАНТ РЕШЕНИЯ - НЕИДЕАЛЬНЫЙ

df1 = df.iloc [:17]
df2 = df.iloc [17:]
# print(df1, df2)

df.plot(x='dtd', y='price', kind='scatter', marker='x', color='green', s=100)

theta_0 = 1
theta_1 = 22
theta_2 = 1100

m = len(df2)
print(df2['price'].max())

x = np.linspace(31, 100, 33)
y = -0.9*x**2 + 102*x - 1056
# x = np.linspace(3, 25, 17)
# y = 100*np.sin(x) + 15*x + 800
x1 = np.linspace(3, 31, 17)
y1 = 1.1*x1**2 - 32*x1 + 1200

df2['prediction'] = -theta_0 * (df2['dtd']**2) + theta_1 * df2['dtd'] + theta_2

# Построение графика полинома 2 степени (восходящего синуса)
plt.plot(x, y, color='red')
plt.plot(x1, y1, color='red')
# Построение графика предсказания
# plt.plot(df2['dtd'], df2['prediction'], color='blue', linestyle=':', linewidth=3)


mse_loss = 1 / (2 * m) * np.sum((y - df2['price']) ** 2)
print(mse_loss)
plt.show()

"""

