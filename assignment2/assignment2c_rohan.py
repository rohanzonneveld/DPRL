import numpy as np

distribution = np.zeros(91)
distribution[0] = 1
failure_probs = np.arange(0.10, 1.01, 0.01)

for _ in range(100):
    distribution = np.concatenate(([np.sum(distribution * failure_probs)], distribution[:-1] * (1 - failure_probs[:-1])))

print(distribution)
print(np.sum(distribution))