import numpy as np
import pandas as pd
from numpy.linalg import lstsq, cond, norm

# Load and clean the data
data_path = 'Part B/LifeExpectancyData.csv'
data = pd.read_csv(data_path)
selected_data = data[['Life expectancy ', 'Schooling']].rename(columns={'Life expectancy ': 'LifeExpectancy'})
clean_data = selected_data.dropna()
X = clean_data['Schooling'].values
y = clean_data['LifeExpectancy'].values

# Create the design matrix for quadratic terms: [1, x, x^2]
X_quad = np.column_stack((np.ones(X.shape), X, X**2))

# Calculate the condition number of A^TA
ATA = X_quad.T @ X_quad
condition_number = cond(ATA)
print(f"Condition number of A^TA: {condition_number}")

# Solve the least squares problem using Normal Equations
beta_normal = np.linalg.inv(ATA) @ (X_quad.T @ y)

# Adding QR Decomposition
Q, R = np.linalg.qr(X_quad)
beta_qr = np.linalg.solve(R, Q.T @ y)  # More numerically stable than using inverse directly

# Solve the least squares problem using numpy's lstsq for comparison
beta_lstsq, residuals, rank, s = lstsq(X_quad, y, rcond=None)

# Calculate relative errors for both methods
error_normal = norm(beta_normal - beta_lstsq) / norm(beta_lstsq)
error_qr = norm(beta_qr - beta_lstsq) / norm(beta_lstsq)

print(f"Relative error for Normal Equations solution: {error_normal}")
print(f"Relative error for QR Decomposition solution: {error_qr}")

# The theoretical maximum relative error based on the condition number (very rough estimate)
max_theoretical_error = condition_number * np.finfo(float).eps
print(f"Maximum theoretical relative error due to condition number: {max_theoretical_error}")
