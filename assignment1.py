import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(42)

T = 10
W = 10

# calculate value function
V = np.zeros((T + 1, W + 1)) # initialize value function
for t in range(T-1,-1,-1): # loop backwards in time
    for w in range(T-t,W+1):
        V[w,t] = max(0.1 * np.sum(V[:w, t+1] + np.ones(w)), V[w, t+1])
        if W - w == t:
            V[w,:t] = V[w,t]

# calculate optimal policy
policy = np.zeros((T + 1, W + 1))
for t in range(T-1,-1,-1):
    for w in range(W+1):
        # find index of first value that is bigger than scalar
        policy[w,t] = w - np.argmax(V[:,t+1] + 1 >= V[w,t+1])

# rotate and plot V matrix
V = np.round(V, 2)
V_rot = np.flipud(V) # flip matrix upside down to make interpretation more intuitive

plt.imshow(V_rot, cmap='viridis', interpolation='nearest')

for i in range(V_rot.shape[0]):
    for j in range(V_rot.shape[1]):
        plt.text(j, i, str(V_rot[i, j]), ha='center', va='center', color='w')

plt.colorbar()
plt.axis('on')
plt.xlabel('Time')
plt.ylabel('Weight in the knapsack')
# plt.savefig('V.png')
plt.show()

# rotate and plot policy
policy = policy.astype(int)
policy_rot = np.flipud(policy)

plt.imshow(policy_rot, cmap='viridis', interpolation='nearest')

for i in range(policy_rot.shape[0]):
    for j in range(policy_rot.shape[1]):
        plt.text(j, i, str(policy_rot[i, j]), ha='center', va='center', color='w')

plt.colorbar()
plt.axis('on')
plt.xlabel('Time')
plt.ylabel('Weight in the knapsack')
# plt.savefig('policy.png')
plt.show()

# Simulation
scores = []
for i in range(1000):
    weight = 0
    score = 0
    for t in range(T):
        w = random.randint(1, W)
        if w <= policy[10-weight,t]:
            weight += w
            score += 1
        if weight == 10:
            break
    scores.append(score)

# Plot and print results of simulation
print(f"The highest score that was achieved was {max(scores)}")
print(f"The lowest score that was achieved was {min(scores)}")
print(f"The average score was {np.mean(scores)}")

plt.hist(scores, bins = [1,2,3,4,5,6,7])
plt.xlabel('Score')
plt.ylabel('Number of occurences')
# plt.savefig("scores_histogram.png")
plt.show()

