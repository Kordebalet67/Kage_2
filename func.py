import pandas as pd
import numpy as np

# dataframe с данными
df = pd.read_csv('kaggleton_2_data.csv')
df.columns = ['x','y']

# Нахождение функции по значениям
coefficients = np.polyfit(df['x'], df['y'], 64)
function = np.poly1d(coefficients)
print('Найденная функция:', function)