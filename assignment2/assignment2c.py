import numpy as np

distribution = np.zeros(91)
distribution[0] = 1
x = np.arange(0.10, 1.01, 0.01) # states are failure probabilities

for _ in range(100):
    distribution = np.concatenate(([np.sum(distribution * x)], distribution[:-1] * (1 - x[:-1]))) # pi_i+1

print(f'pi* = {distribution}')
print(f'sum(pi*) = {np.sum(distribution)}')