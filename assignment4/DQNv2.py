import numpy as np
import tensorflow as tf

import numpy as np
import random
from collections import deque
from plot import plot_Q

class MiniMazeEnvironment:
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.num_states = grid_size * grid_size
        self.num_actions = 4
        self.current_state = None
        self.is_terminal = False
        self.goal_state = (0, grid_size - 1)  # Reward at top-right corner
        self.reward = 0

    def reset(self):
        self.current_state = (self.grid_size - 1, 0)  # Start from the bottom-left corner
        self.is_terminal = False
        self.reward = 0
        return self.current_state

    def step(self, action):
        if self.is_terminal:
            # If the agent is already in a terminal state, no further action is allowed
            return self.current_state, self.reward, self.is_terminal

        next_state = self._get_next_state(action)
        self.current_state = next_state

        if next_state == self.goal_state:
            # Agent reached the goal
            self.is_terminal = True
            self.reward = 1
        
        return next_state, self.reward, self.is_terminal
        

    def _get_next_state(self, action):
        x, y = self.current_state
        if action == 0:  # Move up
            x = max(0, x - 1)
        elif action == 1:  # Move down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Move left
            y = max(0, y - 1)
        elif action == 3:  # Move right
            y = min(self.grid_size - 1, y + 1)

        return (x, y)
    
    def one_hot_encode_state(self, state):
        return tf.convert_to_tensor(state)#tf.one_hot(state[0] * self.grid_size + state[1], depth=self.num_states, dtype=tf.int8)

class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=123)  
        self.dense1 = tf.keras.layers.Dense(24, activation='relu', input_shape=(state_size,), kernel_initializer=initializer, bias_initializer=initializer)
        self.dense2 = tf.keras.layers.Dense(24, activation='relu', kernel_initializer=initializer, bias_initializer=initializer)
        self.output_layer = tf.keras.layers.Dense(action_size, activation='linear', kernel_initializer=initializer, bias_initializer=initializer)

    def call(self, state):
        x = self.dense1(tf.expand_dims(state, axis=0))
        x = self.dense2(x)
        return self.output_layer(x)

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, alpha=0.001, batch_size=32, replay_memory_size=1000, target_update_frequency=10):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.target_update_frequency = target_update_frequency

        self.model = QNetwork(state_size, action_size)
        self.model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.alpha))
        self.model.call(tf.constant(np.zeros((1, state_size)), dtype=tf.int8)) # dummy call to initialize weights

        self.target_model = QNetwork(state_size, action_size)
        self.target_model.compile(loss='mse', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.alpha))
        self.target_model.call(tf.constant(np.zeros((1, state_size)), dtype=tf.int8)) # dummy call to initialize weights
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.replay_memory_size)

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.call(tf.convert_to_tensor(state))
            return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state):
        self.replay_memory.append((state, action, reward, next_state))

    def replay(self):
        if len(self.replay_memory) < self.batch_size:
            return

        minibatch = random.sample(list(self.replay_memory)[:-1], self.batch_size - 1)
        minibatch.append(self.replay_memory[-1]) # Always select last because that is the only move that leads to a reward (necesarry so that the reward can be backpropagated through the q-values)
        states, targets = [], []

        for state, action, reward, next_state in minibatch:
            target = np.array(self.target_model.call(tf.convert_to_tensor(state)))[0]
            next_q_values = self.target_model.call(tf.convert_to_tensor(next_state))
            target[action] = reward + self.gamma * np.max(next_q_values[0])

            states.append(tf.convert_to_tensor(state))
            targets.append(target)

        states = tf.convert_to_tensor(np.array(states), dtype=tf.int8)
        targets = tf.convert_to_tensor(np.array(targets), dtype=tf.float32)

        self.model.train_on_batch(states, targets)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            steps = 0

            while True:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                steps += 1

                self.remember(state, action, reward, next_state)
                self.replay()

                state = next_state

                if done:
                    break

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            if episode % self.target_update_frequency == 0:
                self.update_target_model()

            print(f"Episode: {episode + 1}, #steps: {steps}")

# set up environment
env = MiniMazeEnvironment(grid_size=5)
state_size = 2 # x and y coordinates
action_size = env.num_actions

# train agent
agent = DQNAgent(state_size, action_size, target_update_frequency=5)
agent.train(num_episodes=50)

# extract policy from Q-values
Q = np.empty((5, 5, 4))
policy = np.empty((5, 5), dtype=np.int8)
for i in range(5):
    for j in range(5):
        Q[i, j, :] = agent.target_model.call(tf.convert_to_tensor((i, j)))
        policy[i, j] = np.argmax(Q[i, j, :])

print('\nOptimal policy:\n')
print(policy)
plot_Q(Q)