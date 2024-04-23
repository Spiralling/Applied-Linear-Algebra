import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Load the data
data_path = 'Part B/LifeExpectancyData.csv'  # Updated path for consistency
data = pd.read_csv(data_path)

# Selecting the relevant columns and renaming for convenience
selected_data = data[['Life expectancy ', 'Schooling']].rename(columns={'Life expectancy ': 'LifeExpectancy'})

# Removing rows with NaN values in either 'LifeExpectancy' or 'Schooling'
clean_data = selected_data.dropna()

# Extract independent and dependent variables
X = clean_data['Schooling'].values
y = clean_data['LifeExpectancy'].values

def Normal_Equations(X, y):
    # Create the design matrix for quadratic terms: [1, x, x^2]
    X_quad = np.column_stack((np.ones(X.shape), X, X**2))
    # Solving the normal equations
    beta = np.linalg.inv(X_quad.T @ X_quad) @ X_quad.T @ y
    return beta

def QR_Decomposition(X, y):
    # Create the design matrix for quadratic terms: [1, x, x^2]
    X_quad = np.column_stack((np.ones(X.shape), X, X**2))
    # QR decomposition
    Q, R = np.linalg.qr(X_quad)
    # Solving for beta using QR decomposition
    beta = np.linalg.inv(R) @ Q.T @ y
    return beta

# Solve for coefficients using both methods
beta_normal = Normal_Equations(X, y)
beta_qr = QR_Decomposition(X, y)

# Generate x values for prediction
x_values = np.linspace(X.min(), X.max(), 300)
y_pred_normal = np.polyval(beta_normal[::-1], x_values)
y_pred_qr = np.polyval(beta_qr[::-1], x_values)

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(X, y, alpha=0.5, label='Actual Data')
plt.plot(x_values, y_pred_normal, label='Normal Equations Fit', linestyle='--')
plt.plot(x_values, y_pred_qr, label='QR Decomposition Fit', linestyle='-.')
plt.title('Quadratic Regression: Life Expectancy vs. Schooling')
plt.xlabel('Schooling (Years)')
plt.ylabel('Life Expectancy (Years)')
plt.legend()
plt.grid(True)
plt.show()
