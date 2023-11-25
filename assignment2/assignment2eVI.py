import numpy as np

V = np.zeros(91)
x = np.arange(0.1, 1.01, 0.01)

while True:
    Q = np.zeros((91,2))
    Q[:,0] = x * (1+V[0]) + (1-x) * np.concatenate([V[1:], [V[0]]]) # action 1
    Q[:,1] = 0.6 + V[0] # action 2
    V_new = np.min(Q, axis=1) 

    if np.allclose(V, V_new, atol=1e-6):
        break
    V = V_new

policy = np.argmin(Q, axis=1) + 1

print(f"policy = {policy}")
print(f"V = {V-V[0]}")


