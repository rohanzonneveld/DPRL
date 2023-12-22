import numpy as np


# Maze dimensions
ROWS = 3
COLS = 3

# Actions: Up, Down, Left, Right
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
NUM_ACTIONS = len(ACTIONS)

# Define rewards and terminal state
REWARDS = np.zeros((ROWS, COLS))
REWARDS[0, COLS - 1] = 1  # Reward at the goal state
TERMINAL_STATE = (0, COLS - 1)

# Initialize the policy and value function (Q-values)
policy = np.random.choice(ACTIONS, size=(ROWS, COLS))  # Random initial policy
Q_TABLE = np.zeros((ROWS, COLS, NUM_ACTIONS))

# Parameters
DISCOUNT_FACTOR = 0.9
MAX_ITERATIONS = 100

def take_action(state, action):
    if action == 'UP':
        state = (state[0] - 1, state[1])
    elif action == 'DOWN':
        state = (state[0] + 1, state[1])
    elif action == 'LEFT':
        state = (state[0], state[1] - 1)
    elif action == 'RIGHT':
        state = (state[0], state[1] + 1)

    # cap state to be within the grid
    state = (max(0, min(state[0], ROWS-1)), max(0, min(state[1], COLS-1)))

    return state

# Policy Evaluation
def policy_iteration():
    for _ in range(MAX_ITERATIONS):
        for row in range(ROWS):
            for col in range(COLS):
                state = (row, col)
                if state == TERMINAL_STATE:
                    continue

                # Calculate the expected value under the current policy
                for a in ACTIONS:
                    current_q = Q_TABLE[row, col, ACTIONS.index(a)]
                    next_state = take_action(state, a)
                    
                    reward = REWARDS[next_state[0], next_state[1]]
                    next_q = np.max(Q_TABLE[next_state[0], next_state[1]])
                    additional_value = (reward + DISCOUNT_FACTOR * next_q) - current_q
                
                    # Update Q-value
                    Q_TABLE[row, col, ACTIONS.index(a)] += additional_value
    
        if converged():
            break

# Policy Improvement
def converged():
    policy_stable = True
    for row in range(ROWS):
        for col in range(COLS):
            state = (row, col)
            best_action = None
            best_value = -float('inf')
            for a in ACTIONS:
                value = Q_TABLE[row, col, ACTIONS.index(a)]
                if value > best_value:
                    best_value = value
                    best_action = a
            if best_action != policy[row, col]:
                policy_stable = False
                policy[row, col] = best_action
    
    return policy_stable

policy_iteration()

# Print the resulting optimal policy
print("Optimal Policy:")
print(policy)
print("\nQ-table")
print(Q_TABLE)