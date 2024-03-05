# Classes, methods, and functions to generate DQN-based RL elements

import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
from RL_Actions import *
from gym.spaces import Discrete, Box
from gym import Env
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory # experience replay stack
from collections import deque
import itertools

## Building a custom environment to represent the wireless channel as well as Bob
class Env_Channel_Bob(Env):
    def __init__(self, episode_length, avg_count, n_actions):
        self.action_space = Discrete(n_actions)
        self.observation_space = Box(low=np.array([0,0]), high=np.array([avg_count,16]))
        self.state = np.array([50 + random.randint(-40,40), 8 + random.randint(-7,7)])
        self.episode_length = episode_length
        self.avg_count = avg_count


    def step(self, action):
        self.state, reward = agent_configure(action, self.avg_count)
        self.episode_length -= 1

        # Is the episode over ?
        if self.episode_length <= 0:
            done = True
        else:
            done = False

        # Setting the placeholder for info
        info = {}

        # Returning the step information
        return self.state, reward, done, info
    # Method to reset environment and episode counter
    def reset(self):
        self.state = np.array([50 + random.randint(-40,40), 8 + random.randint(-7,7)])
        self.episode_length = 3 # 100
        return self.state

    def render(self):
        pass
        # Add visualization code later





class DQN_Agent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.005
        self.batch_size = 8 # 32
        self.buffer_size = 20 # 5000
        self.replay_buffer = deque()
        self.Qmodel = Sequential([
            tf.keras.layers.Input(shape=(1, state_size[0])),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='linear'),
            tf.keras.layers.Reshape((action_size,))
        ])
        self.Qmodel.compile(loss="mse",
                      optimizer = Adam(lr=self.lr))

    # Pick action using exploration probability epsilon
    def pick_action(self, current_state):
        if np.random.uniform(0,1) < self.epsilon:
            return np.random.choice(range(self.n_actions))
        Q_vector = self.Qmodel.predict(current_state)[0]
        return np.argmax(Q_vector)

    def epsilon_update(self):
        self.epsilon = self.epsilon * np.exp(-self.epsilon_decay)
        print("New epsilon value: {}".format(self.epsilon))

    # Push current experience to the buffer
    def store_experience(self,current_state, action, reward, next_state, done):
        self.replay_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })

        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.popleft()

    # DQN agent training method
    def train(self):
        np.random.shuffle(self.replay_buffer)
        Batch = list(itertools.islice(self.replay_buffer, 0, self.batch_size))
        for experience in Batch:
            # Q vector of St
            Qt = self.Qmodel.predict(experience["current_state"])
            # Bellman update equation
            Q0 = experience["reward"]
            if not experience["done"]:
                Q = Q0 + self.gamma * np.max(self.Qmodel.predict(experience["next_state"])[0])
            Qt[0][experience["action"]] = Q
            self.Qmodel.fit(experience["current_state"], Qt, verbose=0)

# # Generate Q_model using a DNN
# def Gen_Qmodel(state_size, action_size):
#     Qmodel = Sequential()
#     Qmodel.add(tf.keras.layers.Input(shape=(1,state_size[0])))
#     Qmodel.add(tf.keras.layers.Dense(16, activation='relu'))
#     Qmodel.add(tf.keras.layers.Dense(32, activation='relu'))
#     Qmodel.add(tf.keras.layers.Dense(action_size, activation='linear'))
#     Qmodel.add(tf.keras.layers.Reshape((action_size,)))
#     return Qmodel
#
# # Generate RL agent that uses Qmodel to learn for the Env_Channel_Bob environment
# # The agent is compiled to use Epsilon Greedy optimization
#
# def Gen_DQN_agent(Qmodel, action_size):
#     memory = SequentialMemory(limit=5000, window_length=1)
#     policy = EpsGreedyQPolicy()
#     return DQNAgent(model=Qmodel, memory=memory, policy=policy,
#                   nb_actions=action_size, nb_steps_warmup=10, target_model_update=1e-2)
