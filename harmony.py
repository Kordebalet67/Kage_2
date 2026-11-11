import numpy as np
import matplotlib.pyplot as plt

# параметры функции
A = 10  # амплитуда
omega = 2000  # частота
phi = np.pi/4  # начальная фаза

# переменная x
x = np.linspace(1, 101, 101)
n = np.linspace(1, 101, 101)
koef = (-1) ** n
# гармоническая функция
y = A * np.sin(omega * x + phi) + koef *  x ** 4



# построение графика
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()