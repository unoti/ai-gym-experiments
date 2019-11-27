import numpy as np
from .deeprico import QLearningPolicy
from .probabilitybag import ProbabilityBag
from .agent import Agent

RETURN_WEIGHT = 0.1 # Weight to return surprise memories back to experience pool.

class DeepQPrioritized(QLearningPolicy):
    def __init__(self, env, snapshots=100000, **kwargs):
        super().__init__(env, snapshots=snapshots)
        self.snapshots = None # We'll use the probability bag instead
        self.experience = ProbabilityBag(max_size=snapshots)
        self._expected_reward = 0 # Expected rewards from last action taken.
    
    def suggest_action(self, state, epsilon=None):
        if epsilon == None:
            epsilon = self.epsilon
        if np.random.uniform() <= epsilon:
            action = self.env.action_space.sample()
            self._expected_reward = self.model.predict(state)[action]
            return action

        rewards_all = self.model.predict(state) # Rewards for all possible actions
        action = np.argmax(rewards_all) # Index of the action having the highest rewards.
        self._expected_reward = rewards_all[action]
        return action


    def step_completed(self, prev_state, prev_action, reward, state, done):
        self._decay_epsilon()

        # Save this activity for experience replay.
        surprise_level = abs(reward - self._expected_reward)
        entry = (prev_state, prev_action, reward, state, done)
        self.experience.insert_batch([(surprise_level, entry)])

        self.experience_replay()

    def experience_replay(self):
        if len(self.experience) < self.batch_size:
            return

        batch = self.experience.remove_batch(self.batch_size)
        return_batch = [] # [(weight, item)] of items going back into experience pool.
        for item in batch:
            (state, action, reward, state_next, done) = item
            self._learn_step(state, action, reward, state_next, done)
            # Put this experience back in.
            # We don't know the remaining surprise factor because it's no
            # longer rooted in ground truth. So put it back in at a low delta.
            return_batch.append((RETURN_WEIGHT, item))
        self.experience.insert_batch(return_batch)

def deep_priority(env):
    return Agent(env, DeepQPrioritized(env))