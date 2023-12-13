import numpy as np
import random

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


def learnQ(alpha=0.1, epsilon=0.2, gamma=0.9, num_episodes=1000):
    # Initialize Q-values randomly
    Q = np.ones((3, 3, 4)) # TODO: initialize with ones is important!
    actions = ['up', 'down', 'left', 'right']
    initial_state = (2, 0)  
    num_episodes = 1000

    # Set parameters
    alpha = 0.1  # Learning rate
    epsilon = 0.2  # Exploration-exploitation trade-off
    gamma = 0.9  # Discount factor

    # Perform Q-learning updates
    for episode in range(num_episodes):
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

    # Extract policy from Q-values
    policy = np.argmax(Q, axis=2)
    return policy, Q

if __name__ == '__main__':
    policy, Q = learnQ(alpha=0.1, epsilon=0.2, gamma=0.9, num_episodes=1000)
    print(policy)
    print(Q)


