import numpy as np

V = np.zeros(91)
X = [i/100 for i in range(10,101)]
for j in range(10000):
    V_new = np.zeros(91)
    for idx,v in enumerate(V[:-1]):
        keep_going = (X[idx]* (1+V[0])) + ((1-X[idx]) *V[idx+1])
        replace = 0.6 + V[0]
        V_new[idx] = min(replace, keep_going)
    V_new[-1] = min(0.6 + V[0], 1 + V[0])
    V = V_new

policy = []
for idx, v in enumerate(V[:-1]):
    if (0.6 + V[0]) < ((X[idx]* (1+V[0])) + ((1-X[idx]) * V[idx+1])):
        policy.append(2)
    else:
        policy.append(1)
policy.append(policy[-1])

print(policy)


