# do hyperparameter optimization using optuna of epsilon, exploration constant, and gamma

import optuna
import random
import tqdm
from connect4 import *
from MCTS import mcts_search

state0 = np.array([ [-1,  1,  1,  1, -1, -1,  1],
                    [ 0,  1, -1, -1,  1, -1,  0],
                    [ 0,  1,  1,  1, -1,  1,  0],
                    [ 0, -1, -1, -1,  1, -1,  0],
                    [ 0,  1,  1, -1, -1,  1,  0],
                    [ 0,  1, -1, -1,  1, -1,  0]], dtype=np.int8)

def objective(trial):
    epsilon = trial.suggest_float("epsilon", 1e-5, 1e-3)
    exploration_constant = trial.suggest_float("exploration_constant", 1., 5.)
    gamma = trial.suggest_float("gamma", 0.9, 1.0)

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
                action = mcts_search(state, show_results=False, epsilon=epsilon, exploration_constant=exploration_constant, gamma=gamma)
                state = apply_action(state, action, player)
            
            player *= -1
        
        _, reward = is_terminal(state)
        if reward == 1:
            wins += 1
        elif reward == -1:
            losses += 1
            # print("Loss:")
            # print(state)
            # print()
        else:
            draws += 1

    return losses

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print(study.best_params)