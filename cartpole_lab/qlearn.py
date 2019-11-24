import numpy as np
from collections import defaultdict, deque
import random

from .discretizer import Discretizer
from .agent import Agent

class QLearningPolicy:            
    def __init__(self, env, learning_rate=0.001, epsilon_start=1, epsilon_min=0.01, epsilon_decay=0.995,
                 gamma=0.95, memory_size=10000, batch_size=40):
        """epsilon: probability of exploring by selecting a random action
        gamma: Discount factor for future rewards. Between 0 - 1.
        learning_rate: learning rate for the approximator
        """
        self.env = env
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.gamma = gamma # Discount factor for future rewards.
        self.discretizer = Discretizer([(-4, 4), (-4, 4), (-1, 1), (-4, 4)])
        num_actions = self.env.action_space.n # 2
        self.qvalues = defaultdict(lambda: np.zeros(num_actions)) # values: expected rewards per action index
        self.history = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def suggest_action(self, observation, epsilon=None):
        if epsilon == None:
            epsilon = self.epsilon
        if np.random.uniform() <= epsilon:
            return self.env.action_space.sample()

        state = self.discretizer.get(observation)
        expected_rewards = self.qvalues[state] # Rewards for all possible actions
        # If we haven't seen this state before, select randomly, because argmax returns first always
        if len(np.nonzero(expected_rewards)) == 0:
            return self.env.action_space.sample()
        action = np.argmax(expected_rewards) # Index of the action having the highest rewards.
        return action
    
    def step_completed(self, prev_observation, prev_action, reward, observation, done):
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update Q value
        prev_state = self.discretizer.get(prev_observation)
        state = self.discretizer.get(observation)
        self._sarsa_update(prev_state, prev_action, reward, state, done)

        # Save history
        self.history.append((prev_state, prev_action, reward, state, done))

        self._experience_replay()

    def _sarsa_update(self, prev_state, prev_action, reward, state, done):
        prev_q_all = self.qvalues[prev_state]
        prev_q = prev_q_all[prev_action]
        q_t1_all = self.qvalues[state] # Q values for time t+1
        q_t1 = np.max(q_t1_all) # Expected rewards for taking the best action in t+1

        # SARSA update function Sutton and Barto formula 6.7 p. 130
        prev_q = prev_q + self.learning_rate * (reward + self.gamma * q_t1 - prev_q)
        prev_q_all[prev_action] = prev_q
    
    def _experience_replay(self):
        if (len(self.history) < self.batch_size):
            return
        steps = random.sample(self.history, self.batch_size)
        for (prev_state, prev_action, reward, state, done) in steps:
            self._sarsa_update(prev_state, prev_action, reward, state, done)
    
    def episode_completed(self, episode):
        pass
    
    def peek(self, state=None):
        """Returns estimated sum of rewards for the given state. If no state is provided,
        we'll reset the environment and use the initial state.
        """
        if not state:
            state = self.env.reset()
        state = self.discretizer.get(state)
        return self.qvalues[state]

def q_learning(env):
    return Agent(env, QLearningPolicy(env))