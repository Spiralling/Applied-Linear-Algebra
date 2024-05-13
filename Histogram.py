import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq, norm

# Load and clean the data
data_path = 'Part B/LifeExpectancyData.csv'
data = pd.read_csv(data_path)
selected_data = data[['Life expectancy ', 'Schooling']].rename(columns={'Life expectancy ': 'LifeExpectancy'})
clean_data = selected_data.dropna()
X = clean_data['Schooling'].values
y = clean_data['LifeExpectancy'].values

# Design matrix for quadratic terms: [1, x, x^2]
X_quad = np.column_stack((np.ones(X.shape), X, X**2))

# Solve using Normal Equations
beta_normal = np.linalg.inv(X_quad.T @ X_quad) @ (X_quad.T @ y)
# Solve using QR Decomposition
Q, R = np.linalg.qr(X_quad)
beta_qr = np.linalg.solve(R, Q.T @ y)

# Calculate residuals
y_pred_normal = X_quad @ beta_normal
residuals_normal = y - y_pred_normal

y_pred_qr = X_quad @ beta_qr
residuals_qr = y - y_pred_qr

# Plotting histograms of residuals
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.hist(residuals_normal, bins=200, color='blue', alpha=0.7)
plt.title('Histogram of Residuals (Normal Equations)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(residuals_qr, bins=200, color='green', alpha=0.7)
plt.title('Histogram of Residuals (QR Decomposition)')
plt.xlabel('Residuals')

plt.tight_layout()
plt.show()
