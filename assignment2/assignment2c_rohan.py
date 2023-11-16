import numpy as np

distribtion = np.zeros(91)
distribtion[0] = 1
failure_probs = np.arange(0.10, 1.01, 0.01)

for _ in range(100):
    pi_1_new = np.sum(distribtion * failure_probs)
    pi_rest_new = distribtion[:-1] * (1 - failure_probs[:-1])
    distribtion[0] = pi_1_new
    distribtion[1:] = pi_rest_new

print(distribtion)
print(np.sum(distribtion))