import numpy as np
import random
from collections import deque


def take_action(state, action):
    if action == 'up':
        state = (state[0] - 1, state[1])
    elif action == 'down':
        state = (state[0] + 1, state[1])
    elif action == 'left':
        state = (state[0], state[1] - 1)
    elif action == 'right':
        state = (state[0], state[1] + 1)

    # cap state to be within the grid
    state = (max(0, min(state[0], 2)), max(0, min(state[1], 2)))
    reward = 1 if state == (0,2) else 0

    return state, reward

def converged(cache):
    if len(cache) < 2:
        return False
    else:
        return np.linalg.norm(cache[0] - cache[1]) < 1e-20

def learnQ(alpha=0.1, epsilon=1, gamma=0.9):
    # Initialize Q-values randomly
    Q = np.zeros((3, 3, 4)) # TODO: initialize with ones is important!
    cache = deque(maxlen=2)
    actions = ['up', 'down', 'left', 'right']
    initial_state = (2, 0)  

    # Perform Q-learning updates
    while not converged(cache):
        current_state = initial_state
        terminal_state = False
        while not terminal_state:
            # Choose action using epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = random.choice(range(4))
            else:
                action = np.argmax(Q[current_state[0], current_state[1], :])
            
            # Take the chosen action and observe the next state and reward
            next_state, reward = take_action(current_state, actions[action])
            if reward == 1: terminal_state = True
            
            # Q-learning update
            Q[current_state[0], current_state[1], action] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[current_state[0], current_state[1], action])
            
            # Move to the next state
            current_state = next_state
        cache.append(Q.copy())

    # Extract policy from Q-values
    policy = np.argmax(Q, axis=2)
    return policy, Q

if __name__ == '__main__':
    # random.seed(42)
    # np.random.seed(42)
    # num = 0
    # alphas = [i * 0.1 for i in range(1, 11)]
    # epsilons = [i * 0.1 for i in range(1, 11)]
    # for alpha in alphas:
    #     for epsilon in epsilons:
    #         policy, Q = learnQ(alpha=alpha, epsilon=epsilon, gamma=0.9)
    #         reward = calculate_reward(policy)
    #         print(f"alpha = {alpha:.1f}, epsilon = {epsilon:.1f}, reward = {reward}")
    #         if reward < 8:
    #             num += 1
    # print(num)

    policy, Q = learnQ()
    print(policy)
    print(Q)


