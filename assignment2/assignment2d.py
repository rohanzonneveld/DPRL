import numpy as np

# construct state vector
x = np.arange(0.1, 1.01, 0.01)

# create coefficient for phi in Poisson equation
coeffs = np.zeros(92) # create matrix with extra zero to start with
for i in range(1,92):
    coeffs[i] = (1-x[-i])*coeffs[i-1] - 1
coeffs = np.delete(coeffs, 0) # delete extra zero

# create A and b matrix
A = np.eye(91) # identity as V(2) appears in equation 2 and so on
A[:,0] = -coeffs[::-1] # replace first column with coeffs for phi since we replace V(1) with phi
b = np.ones(91)

# solve for V an phi
V = np.linalg.solve(A, b)
phi = V[0]
V[0] = 0

print(f'V = {V}')
print(f'phi = {phi}')


