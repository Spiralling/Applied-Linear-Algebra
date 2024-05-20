import numpy as np

# Define the matrix A
A = np.array([
    [5, 5, 0, 4],
    [1, 1, 5, 0],
    [3, 2, 0, 4],
    [5, 3, 0, 5],
    [0, 0, 4, 0]
])

#Singular Value Decomposition
[U, S, VT] = np.linalg.svd(A)

print("Matrix U (Left Singular Vectors):")
print(U)
print("\nSingular Values (Sigma):")
print(np.diag(S))  # Converting the singular values into a diagonal matrix for display purposes
print("\nMatrix V^T (Right Singular Vectors):")
print(VT)
