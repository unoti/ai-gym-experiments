from .progress import log_progress
from .charts import plot_episode_scores
import numpy as np
import time

class Agent:
    """
    A reinforcement learning agent. Provides infrastructue to support
    training.  The actual intelligence is injected into the agent via the policy.
    """
    def __init__(self, env, policy):
        """
        @param env: The environment.
        @param policy: The "brain" of the agent with the following methods and attributes:
            step_completed(state, action, reward, state_next, done)
                Called when we've completed one step.
            epsilon: A number from 0-1 indicating the probability of taking a random action.
            episode_completed(episode):
                Called when we have completed training one episode.
        """
        self.env = env
        self.policy = policy
    
    def run_episode(self, render=False):
        episode = [] # prev_state, prev_action, reward, state, done
        state = self.env.reset()
        action = None
        while True:
            action = self.policy.suggest_action(state)
            state_next, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward, state_next, done))
            self.policy.step_completed(state, action, reward, state_next, done)
            state = state_next
            if render:
                self.env.render()
            if done:
                break
        return episode
    
    def train(self, episodes=100, render=False, epsilon_start=1.0, epsilon_min = 0.01, epsilon_decay=0.995):
        episode_scores = []
        epsilons = []
        image_update_seconds = 5
        last_update_time = time.time()
        
        epsilon_orig = self.policy.epsilon # Save the original epsilon, in case we want to do external things with agent later.
        self.policy.epsilon = epsilon_start
        
        chartfile = r'images/progress.png'
        for _ in log_progress(range(episodes), name='Episodes'):
            episode = self.run_episode(render=render)
            episode_scores.append(total_rewards(episode))
            epsilons.append(self.policy.epsilon)
            self.policy.episode_completed(episode)
            now = time.time()
            if now - last_update_time > image_update_seconds:
                plot_episode_scores(episode_scores, epsilons, save=chartfile)
                last_update_time = now

        plot_episode_scores(episode_scores, epsilons, save=chartfile)
        
        self.policy.epsilon = epsilon_orig # Restore epsilon.
    
    def demo(self, episodes=10, render=True):
        """
        Run some sessions without any random actions; always use the policy.
        """
        scores = []
        orig_epsilon = self.policy.epsilon
        self.policy.epsilon = 0
        for _ in range(episodes):
            episode = self.run_episode(render=render)
            rewards = total_rewards(episode)
            scores.append(rewards)
            print('score=',rewards)
        print('Avg rewards=',np.mean(scores))
        self.policy.epsilon = orig_epsilon

def total_rewards(episode):
    return sum([reward for (state, action, reward, next_state, done) in episode])