from connect4 import *
from MCTS import mcts_search

import random
import tqdm

state0 = np.array([ [-1,  1,  1,  1, -1, -1,  1],
                    [ 0,  1, -1, -1,  1, -1,  0],
                    [ 0,  1,  1,  1, -1,  1,  0],
                    [ 0, -1, -1, -1,  1, -1,  0],
                    [ 0,  1,  1, -1, -1,  1,  0],
                    [ 0,  1, -1, -1,  1, -1,  0]], dtype=np.int8)

wins = 0
draws = 0
losses = 0


for _ in tqdm.tqdm(range(100)):
    state = state0.copy()
    player = 1
    while not is_terminal(state)[0]:
        if player == -1:
            action = random.choice(get_actions(state))
            state = apply_action(state, action, player)
        else:
            action = mcts_search(state, show_results=False, epsilon=1e-4, exploration_constant=2., gamma=0.9)
            state = apply_action(state, action, player)
        
        player *= -1
    
    _, reward = is_terminal(state)
    if reward == 1:
        wins += 1
    elif reward == -1:
        losses += 1
        print("Loss:")
        print(state)
        print()
    else:
        draws += 1

print(f"Wins: {wins}")
print(f"Draws: {draws}")
print(f"Losses: {losses}")