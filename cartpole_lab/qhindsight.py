import numpy as np
from collections import defaultdict, deque
import random

from .discretizer import Discretizer
from .agent import Agent

class QHindsightPolicy:
    """The idea of this policy is to use full hindsight of each episode.
    For each state we will store the actual total rewards for the episode
    when taking a given action in that state.
    """
    def __init__(self, env, learning_rate=0.001, epsilon_start=1, epsilon_min=0.01, epsilon_decay=0.995):
        """epsilon: probability of exploring by selecting a random action
        gamma: Discount factor for future rewards. Between 0 - 1.
        learning_rate: learning rate for the approximator
        """
        self.env = env
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.discretizer = Discretizer([(-4, 4), (-4, 4), (-1, 1), (-4, 4)])
        num_actions = self.env.action_space.n # 2
        self.qvalues = defaultdict(lambda: np.zeros(num_actions)) # values: expected rewards per action index

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
        #print('action=%s state=%s all_rew=%s' % (action, state, expected_rewards))
        return action
    
    def step_completed(self, prev_observation, prev_action, reward, observation, done):
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def episode_completed(self, episode):
        total_rewards = len(episode) # For other environments we'd actually sum rewards, but this works here.
        for (observation, action, reward, next_obs, done) in episode: # Saved in agent.py run_episode()
            if action == None:  # When we're presented with the first state we have no action.
                continue
            state = self.discretizer.get(observation) # Convert continuous observation into discrete state.
            q_all = self.qvalues[state] # Q values for all actions in this state.
            existing_q_for_action = q_all[action]
            # Adjust the expected total rewards for this action towards total_rewards.
            delta_q = total_rewards - existing_q_for_action
            q_all[action] = existing_q_for_action + self.learning_rate * delta_q
    
    def peek(self, state=None):
        """Returns estimated sum of rewards for the given state. If no state is provided,
        we'll reset the environment and use the initial state.
        """
        if not state:
            state = self.env.reset()
        state = self.discretizer.get(state)
        return self.qvalues[state]

def q_hindsight_agent(env):
    return Agent(env, QHindsightPolicy(env))