import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.DataFrame({
    'size_m2': [49, 52, 53, 98, 100, 120, 140, 160, 180, 200, 230],
    'price': [50, 60, 110, 115, 150, 145, 160, 150, 160, 165, 140]
})


theta_0 = 0
theta_1 = 1

m = len(df)

df['size_m2_normalised'] = (df['size_m2'] - df['size_m2'].mean()) / (df['size_m2'].max() - df['size_m2'].min())
df['size_m2_normalised'].describe()

theta_0 = 0.0
theta_1 = 0.0
theta_2 = 0.0
theta_3 = 0.0

n_iterations = 10_000
alpha = 1

loss_values = []
intermediate_models = []

for i in range(n_iterations):
    predictions = (theta_0 + 
                   theta_1 * df['size_m2_normalised'] + 
                   theta_2 * (df['size_m2_normalised']**2) +
                   theta_3 * (df['size_m2_normalised']**3))
    
    loss = 1 / (2 * m) * ((predictions - df['price']) ** 2).sum()
    loss_values.append(loss)
    
    if i % 100 == 0:
        print(f'Iteration: {i}, loss: {loss}')
    
    gradient_theta_0 = (1 / m) * (predictions - df['price']).sum()
    gradient_theta_1 = (1 / m) * np.sum((predictions - df['price']) * df['size_m2_normalised'])
    gradient_theta_2 = (1 / m) * np.sum((predictions - df['price']) * (df['size_m2_normalised']**2))
    gradient_theta_3 = (1 / m) * np.sum((predictions - df['price']) * (df['size_m2_normalised']**3))
    
    theta_0 = theta_0 - alpha * gradient_theta_0
    theta_1 = theta_1 - alpha * gradient_theta_1
    theta_2 = theta_2 - alpha * gradient_theta_2
    theta_3 = theta_3 - alpha * gradient_theta_3
    
    intermediate_models.append((theta_0, theta_1, theta_2, theta_3))

plt.figure(figsize=(10, 6))
# plt.plot(range(1, n_iterations+1), loss_values)
plt.plot(range(20, n_iterations+1), loss_values[19:])
plt.xlabel('Iterations')
plt.ylabel('Loss')



plt.figure(figsize=(10, 6))
plt.scatter(df['size_m2'], df['price'], marker='x', color='red', s=100, label='Actual')

for i, model in enumerate(intermediate_models):
    predictions = (model[0] + 
                   model[1] * df['size_m2_normalised'] + 
                   model[2] * (df['size_m2_normalised']**2) + 
                   model[3] * (df['size_m2_normalised']**3))
    
    plt.plot(df['size_m2'], predictions, color='blue', linestyle='-', linewidth=0.5)

plt.show()