import optuna

from tabularQlearning import learnQ

def calculate_reward(policy):
    reward = 0
    if policy[2,0] == 0 or policy[2,0] == 3: reward += 1
    if policy[2,1] == 0 or policy[2,1] == 3: reward += 1
    if policy[2,2] == 0: reward += 1
    if policy[1,0] == 0 or policy[1,0] == 3: reward += 1
    if policy[1,1] == 0 or policy[1,1] == 3: reward += 1
    if policy[1,2] == 0: reward += 1
    if policy[0,0] == 3: reward += 1
    if policy[0,1] == 3: reward += 1

    return reward

    

def objective(trial):
    alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
    epsilon = trial.suggest_uniform('epsilon', 0.0, 1.0)
    num_episodes = trial.suggest_int('num_episodes', 0, 1000)
    rewards = []
    for i in range(10):
        policy = learnQ(alpha=alpha, epsilon=epsilon, gamma=0.9, num_episodes=num_episodes)
        reward = calculate_reward(policy)
        rewards.append(reward)
    reward = sum(rewards) / len(rewards)
    return reward

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=250)
print(study.best_params)