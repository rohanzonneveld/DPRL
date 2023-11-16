import numpy as np

# b: Simulation
distribution = np.zeros(91)                                 # Initialize the distribution with zero's
distribution[0] = 1                                         # The starting point is the first state
failure_probs = [i / 100 for i in range(10, 101)]           # Vector with the failure rate at all different states

for i in range(1,92):
    cumulative_failure = 0                                  # Initialize the new rate of failure 
    for j in range(i, 0, -1):                               # Loop over all non-zero entries in distribution
        prob = (1 - failure_probs[j-1]) * distribution[j-1] # Calculate probability of ending up in next state
        failure = distribution[j-1] - prob                  # Calculate the probability of failing from current state
        if i != 91 or j != i:                               # When probability of failure = 1, don't put zero in the next state, as it doesn't exist. From this state you will always fail
            distribution[j] = prob                          
        cumulative_failure += failure                       # Sum the probabilities of failure from all different states
    distribution[0] = cumulative_failure                    # Replace the old probability of failure


for x in distribution:
    print(x)

print(sum(distribution))
