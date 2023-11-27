import random
from connect4 import *

'''
Only one child gets created for every node since if there is a child the maximum will be selected.
'''


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def mcts_search(state, iterations=1000):
    root = Node(state)

    for i in range(iterations):
        node = root
        # Selection: select a leaf node
        while not is_terminal(node.state)[0] and node.children:
            node = max(node.children, key=lambda n: UCB(node.visits, n.visits, n.value))

        # Expansion: expand the leaf node
        if not is_terminal(node.state)[0]:
            for action in get_actions(node.state):
                new_state = apply_action(node.state.copy(), action, -1)
                leaf = Node(new_state, parent=node)
                node.children.append(leaf)
            node = random.choice(node.children)
            
        # Simulation: 
        while not is_terminal(node.state)[0]:
            action = random.choice(get_actions(node.state))
            node.state = apply_action(node.state, action, 1)
            action = random.choice(get_actions(node.state))
            node.state = apply_action(node.state, action, -1)            

        # Backpropagation
        _, reward = is_terminal(node.state)
        backpropagate(node, reward)

    # Select the action with the highest average value
    return max(root.children, key=lambda n: n.visits).state

def UCB(p_visits, c_visits, c_value, exploration_weight=1.0):
    if c_visits == 0:
        return float('inf')
    return c_value / c_visits + exploration_weight * np.sqrt(np.log(p_visits) / c_visits)

def backpropagate(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

# Play a game of a random agent against MCTS
state = np.zeros((6, 7), dtype=np.int8)
player = 1
while not is_terminal(state)[0]:
    if player == 1:
        action = random.choice(get_actions(state))
        state = apply_action(state, action, player)
    else:
        state = mcts_search(state)
    player *= -1


