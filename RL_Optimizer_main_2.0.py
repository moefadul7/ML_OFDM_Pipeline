from RL_Agent_Discrete_DQN import Env_Channel_Bob, DQN_Agent
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# DQN & Env settings,
episode_length = 3 #100
avg_count = 100   # Number of transfers over which BLER and Similarity are calculated
n_actions = 36    # Number of discrete actions
n_episodes = 2
nb_steps = 15 #30000

# Path for the DQN weights file
Q_model_Weigths = 'Q_Weigths'

# Sanity check for RL environment:
def check_env(env, episodes=10):
    for episode in range(1, episodes + 1):
        state = env.reset()
        print('initial state is: {}\n\n'.format(state))
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            print("Episode: {}, step number {} , reward is {}".format(episode, env.episode_length, reward))
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))


def main():
    # Initialize an RL-agent environment
    env = Env_Channel_Bob(episode_length, avg_count, n_actions)
    # Test the interaction between environment and agent
    #check_env(env, 10)

    # # Initialize DQN Agent
    agent = DQN_Agent(env.observation_space.shape, env.action_space.n)
    #
    agent.Qmodel.summary()
    counter = 0

    # # Generate & Compile RL agent that uses Qmodel for DQN learning
    #agent.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['mse'])

    # Train the RL agent
    #Hist = agent.fit(env, nb_steps=30000, visualize=False, verbose=1)
    for e in range(n_episodes):
        current_state = env.reset()
        while env.episode_length > 0:
            counter += 1
            # the agent computes the action to perform
            action = agent.pick_action(current_state)
            next_state, reward, done, _ = env.step(action)
            #next_state = np.array([next_state])

            # push experience to replay buffer
            agent.store_experience(current_state, action, reward, next_state, done)

            if done:
                agent.epsilon_update()
                break
            current_state = next_state
        if counter >= agent.batch_size:
            agent.train()

    # Save Q model for exploitation
    agent.Qmodel.save('Q_model.keras')
#
if __name__ == '__main__':
    main()