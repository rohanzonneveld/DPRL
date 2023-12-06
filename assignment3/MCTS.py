import random
import matplotlib.pyplot as plt
from connect4 import *


class Node:
    def __init__(self, state, parent=None, player=1, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.wins = 0
        self.action = action
        
        if self.parent:
            self.player = 1 if self.parent.player == -1 else -1
        else:
            self.player = player

    def rollout(self, gamma=0.99):
        state = self.state.copy()
        player = self.player
        moves = 0
        finished, reward = is_terminal(state)
        while not finished:
            action = random.choice(get_actions(state))
            state = apply_action(state, action, player)
            player *= -1
            moves += 1
            finished, reward = is_terminal(state)

        self.value = (gamma**moves) * reward
        
        return reward

def mcts_search(state, iterations=1000, gamma=0.99):
    root = Node(state, player=1)
    cache = []

    while not converged(cache):
        node = root
        # Selection: select a leaf node
        while not is_terminal(node.state)[0] and node.children:
            if node.player == 1:
                node = max(node.children, key=lambda n: UCB(n))
            else:
                node = min(node.children, key=lambda n: n.visits)

        # Expansion: expand the leaf node
        if not is_terminal(node.state)[0]:
            for action in get_actions(node.state):
                new_state = apply_action(node.state.copy(), action, node.player)
                leaf = Node(new_state, parent=node, action=action)
                node.children.append(leaf)
            node = random.choice(node.children)

        # Simulation:
        reward = node.rollout(gamma)   

        # Backpropagation
        update_nodes(node, reward) # update visits and wins
        backpropagate(node.parent, gamma) # backpropagate the value
        cache.append(root.value)
    visualize_convergence(cache)
    for child in root.children:
        print(f'Action: {child.action+1}, \n Win probability: {child.wins/child.visits} \n')

    # Select the action with the highest average value
    return max(root.children, key=lambda n: n.value).action

def UCB(node, exploration_weight=1.0):
    if node.visits == 0:
        return float('inf')
    return node.wins / node.visits + exploration_weight * np.sqrt(np.log(node.parent.visits) / node.visits)

def update_nodes(node, reward):
    while node:
        node.visits += 1
        if reward == 1:
            node.wins += 1
        node = node.parent

def backpropagate(node, gamma):
    while node:
        values = []
        if node.player == -1:
            # calculate the value according to Q(x,a), use max value of grandchildren since child is players action
            for child in node.children:
                try:
                    max_value_children = max(child.children, key=lambda n: n.value).value
                except:
                    max_value_children = 0

                values.append(child.visits *(gamma*max_value_children))
        else:
            # calculate the value according to Q(x,a), use mean value of children since child is opponents (random) action
            for child in node.children:

                mean_value_grandchildren = np.mean([grandchild.value for grandchild in child.children])
                if np.isnan(mean_value_grandchildren): mean_value_grandchildren = 0
                values.append(child.visits * (gamma*mean_value_grandchildren))
        
        node.value = np.sum(values)/node.visits
        node = node.parent

def transition(state, action):
    # apply the action of the player to the state
    state = apply_action(state, action, 1)

    # check if the game is over
    if is_terminal(state)[0]:
        return state
    
    # apply the random action of the opponent to the state
    opponents_action = random.choice(get_actions(state))
    state = apply_action(state, opponents_action, -1)

    return state

def converged(cache, window_size=10, tolerance=1e-6):
    cache = np.array(cache)

    if cache.shape[0] < window_size:
        return False

    recent_values = cache[-window_size:-1]
    recent_mean = np.mean(recent_values)

    return abs(cache[-1] - recent_mean) < tolerance

def visualize_board(board):
    cmap = plt.get_cmap('tab20')  # Color map for players

    # Create a plot and set the size of the figure
    plt.figure(figsize=(7, 6))

    # Draw the board
    plt.imshow(board, cmap=cmap)

    # Show player discs
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 1:
                plt.scatter(j, i, s=400, marker='o', c='blue', edgecolors='black')
            elif board[i, j] == -1:
                plt.scatter(j, i, s=400, marker='o', c='green', edgecolors='black')

    # plt.gca().invert_yaxis()  # Invert the y-axis to match the board orientation
    plt.axis('off')  # Turn off axis labels
    plt.show() 

def visualize_convergence(values):
    plt.figure()
    plt.plot(values)
    plt.show()

# Play a game of a random agent against MCTS
# state = np.zeros((6, 7), dtype=np.int8)
state = np.array([[ -1, 1,  1,  1, -1, -1,  1],
                  [ 0,  1, -1, -1,  1, -1,  0],
                  [ 0,  1,  1,  1, -1,  1,  0],
                  [ 0, -1, -1, -1,  1, -1,  0],
                  [ 0,  1,  1, -1, -1,  1,  0],
                  [ 0,  1, -1, -1,  1, -1,  0]], dtype=np.int8)
player = 1
while not is_terminal(state)[0]:
    if player == -1:
        action = random.choice(get_actions(state))
        # action = int(input('Enter action: ')) - 1
        state = apply_action(state, action, player)
    else:
        action = mcts_search(state)
        state = apply_action(state, action, player)
    print('\n\n')
    # print(np.flipud(state))
    visualize_board(np.flipud(state))
    
    player *= -1

if is_terminal(state)[1] == 1:
    print('MCTS won!')
elif is_terminal(state)[1] == -1:
    print('Random agent won!')
else:
    print('Draw!')



