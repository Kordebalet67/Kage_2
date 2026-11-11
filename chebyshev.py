import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.polynomial.chebyshev import Chebyshev
import torch

df = pd.read_csv('kaggleton_2_data.csv')
# df.plot(x='dtd', y='price', kind='scatter', marker='x', color='green', s=100)
df.columns = ['x','y']
m = len(df)
print(df)

plt.figure(figsize=(10,7))
for i in [16,32,64]:
    c = Chebyshev.fit(df.x, df.y, i)
    df[f'y{i}'] = c(df.x)
sns.lineplot(data=df.drop(columns='x'))


print(df)

mse_loss = 1 / (2 * m) * np.sum((df['y32'] - df['y']) ** 2)
print(mse_loss)
mse_loss = 1 / (2 * m) * np.sum((df['y64'] - df['y']) ** 2)
print(mse_loss)
plt.show()