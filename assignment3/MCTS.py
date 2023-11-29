import random
from connect4 import *

'''
Notes:
- You want to find the best action of the player to take from a given state
- Therefore connections between states are actions of the player
- Therefore states must always have a state where the player played last
- What to do with the opponent's actions?
- Proposal: every state contains substates for every possible action of the opponent
- A new tree starts from every substate
- The value of a state is the average value of the substates since the probability of the opponent taking a certain action is equal

'''


class Node:
    def __init__(self, state, parent=None, player=-1):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.wins = 0
        if self.parent:
            self.player = 1 if self.parent.player == -1 else 1
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
    root = Node(state, player=-1)

    for i in range(iterations):
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
                new_state = apply_action(node.state.copy(), action, 1)
                leaf = Node(new_state, parent=node)
                node.children.append(leaf)
            node = random.choice(node.children)
            
        # Simulation:
        rollout_state = node.rollout(gamma)   

        # Backpropagation
        _, reward = is_terminal(rollout_state)
        update_nodes(node, reward) # update visits and wins
        backpropagate(node.parent, gamma) # backpropagate the value

    # Select the action with the highest average value
    return max(root.children, key=lambda n: n.value).state

def UCB(node, exploration_weight=1.0):
    if node.visits == 0:
        return float('inf')
    return node.wins / node.visits + exploration_weight * np.sqrt(np.log(node.parent.visits) / node.visits)

def update_nodes(node, reward):
    while node:
        node.visits += 1
        if reward == 1:
            node.wins += reward
        node = node.parent

def backpropagate(node, gamma):
    while node:
        values = []
        if node.player == 1:
            for child in node.children:
                values.append(child.visits *(gamma*max(child.children, key=lambda n: n.value).value))
        else:
            mean_value_children = np.mean([child.value for child in node.children])
            for child in node.children:
                values.append(child.visits * (child.wins + gamma*mean_value_children))
        
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

# Play a game of a random agent against MCTS
state = np.zeros((6, 7), dtype=np.int8)
player = -1
while not is_terminal(state)[0]:
    if player == -1:
        action = random.choice(get_actions(state))
        state = apply_action(state, action, player)
    else:
        state = mcts_search(state)
    print('\n\n')
    print(np.flipud(state))
    
    player *= -1



