import numpy as np

# construct state vector
x = np.arange(0.1, 1.01, 0.01) 

# construct transition matrices for both actions
P1 = np.zeros((91,91))
P2 = np.zeros((91,91))
P2[:,0] = 1
for i in range(90):
    P1[i,0] = x[i]
    P1[i,i+1] = 1 - x[i]
P1[-1,0] = 1

# Policy Iteration
# Step 1: Set policy alpha
alpha = np.ones(91)

# repeat 2 and 3 until convergence
while True:
    # Step 2: Solve Poisson equation
    # Construct vector r and matrix P
    r = np.zeros(91)
    P = np.zeros((91,91))
    for i in range(91):
        if alpha[i] == 1:
            r[i] = x[i]
            P[i,:] = P1[i,:]
        elif alpha[i] == 2:
            r[i] = 0.6
            P[i,:] = P2[i,:]

    # Construct matrix A and vector b
    # V + phi = r + P*V
    # (I-P)*V + phi = r
    A = np.eye(91) - P
    A[:,0] = 1 # we set V(1) to 0 so we can change V(1) to phi in x matrix
    b = r

    # Solve for V and phi
    V = np.linalg.solve(A, b)
    V[0]=0 # change phi back to V(1)

    # Step 3: Improve policy
    Q = np.zeros((91,2))
    Q[:,0] = x + np.dot(P1,V)
    Q[:,1] = 0.6 + np.dot(P2,V)
    alpha_new = np.argmin(Q,axis=1) + 1

    # Check for convergence
    if np.all(alpha_new == alpha):
        break
    else:
        alpha = alpha_new

print(f'policy = {alpha}')
print(f'V = {V}')