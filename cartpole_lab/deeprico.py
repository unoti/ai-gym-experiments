from collections import deque
import random
import numpy as np

from .agent import Agent
from .approximator import Approximator

class QLearningPolicy:            
    def __init__(self, env, learning_rate=0.001, epsilon_start=1, epsilon_min=0.01, epsilon_decay=0.995,
                 gamma=0.95, snapshots=1000000, batch_size=20):
        """epsilon: probability of exploring by selecting a random action
        gamma: Discount factor for future rewards. Between 0 - 1.
        learning_rate: learning rate for the approximator
        """
        self.env = env
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma # Discount factor for future rewards.
        self.snapshots = deque(maxlen=snapshots) # [(prev_state, prev_action, reward, next_state, done)]
        num_inputs = self.env.observation_space.shape[0] # 4
        num_outputs = self.env.action_space.n # 2
        self.model = Approximator(num_inputs, num_outputs, learning_rate)
        self.batch_size = batch_size

    def suggest_action(self, state, epsilon=None):
        if epsilon == None:
            epsilon = self.epsilon
        if np.random.uniform() <= epsilon:
            return self.env.action_space.sample()

        rewards_all = self.model.predict(state) # Rewards for all possible actions
        action = np.argmax(rewards_all) # Index of the action having the highest rewards.
        return action

    def step_completed(self, prev_state, prev_action, reward, state, done):
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Save this activity for experience replay.
        self.snapshots.append((prev_state, prev_action, reward, state, done))
        
        if len(self.snapshots) >= self.batch_size:
            # Replay a batch of experiences.  This shuffles them so we don't train on a bunch of similar
            # states all at once (temporal separation).
            steps = random.sample(self.snapshots, self.batch_size)
            for (prev_state, prev_action, reward, state, done) in steps:
                rewards_all = self.model.predict(prev_state)
                if done:
                    future_rewards = -1
                else:
                    future_rewards = np.max(self.model.predict(state))
                target_reward = reward + self.gamma * future_rewards
                rewards_all[prev_action] = target_reward
                self.model.train(prev_state, rewards_all)
    
    def episode_completed(self, episode):
        pass
    
    def peek(self, state=None):
        """Returns estimated sum of rewards for the given state. If no state is provided,
        we'll reset the environment and use the initial state.
        """
        if not state:
            state = self.env.reset()
        return self.model.predict(state)

def deep_rico(env):
    return Agent(env, QLearningPolicy(env))