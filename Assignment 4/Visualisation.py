import numpy as np
import matplotlib.pyplot as plt

# Matrix U (Left Singular Vectors):
U = np.array([
    [-0.64957922, -0.0414809, -0.70894327, 0.22621394, 0.15017609],
    [-0.11142727, 0.77931786, -0.13714305, -0.24130638, -0.55064567],
    [-0.42776314, -0.06032627, 0.51051713, 0.62639909, -0.40046958],
    [-0.61836908, -0.07152618, 0.41037563, -0.63561117, 0.20023479],
    [-0.01640663, 0.61821336, 0.22261024, 0.30695457, 0.68830708]
])

# Matrix V^T (Right Singular Vectors):
VT = np.array([
    [-0.62769431, -0.49261077, -0.05054067, -0.60064976],
    [0.00518777, 0.0368684, 0.99217874, -0.11914345],
    [-0.05131961, -0.7454196, 0.10674037, 0.65598964],
    [-0.77674915, 0.44757678, 0.04041641, 0.44125084]
])

# Visualize the columns of U
plt.figure(figsize=(10, 6))
for i in range(U.shape[1]):
    plt.plot(U[:, i], marker='o', label=f'Profile {i+1}')
plt.title('Viewer Profiles in U Matrix')
plt.xlabel('Viewer')
plt.ylabel('Profile Strength')
plt.legend()
plt.grid(True)
plt.show()

# Visualize the rows of V^T
plt.figure(figsize=(10, 6))
genres = ['Drama 1', 'Drama 2', 'Scify', 'Documentary']
for i in range(VT.shape[0]):
    plt.bar(genres, VT[i], alpha=0.7, label=f'Movie Profile {i+1}')
plt.title('Movie Profiles in V^T Matrix')
plt.xlabel('Movie Genre')
plt.ylabel('Profile Strength')
plt.legend()
plt.show()
