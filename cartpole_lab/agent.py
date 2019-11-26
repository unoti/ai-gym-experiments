from .progress import log_progress
from .charts import plot_episode_scores
import numpy as np
import time

class Agent:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
    
    def run_episode(self, render=False):
        episode = [] # prev_state, prev_action, reward, state, done
        state = self.env.reset()
        action = None
        while True:
            prev_state = state
            prev_action = action
            action = self.policy.suggest_action(state)
            state, reward, done, _ = self.env.step(action)
            episode.append((prev_state, prev_action, reward, state, done))
            self.policy.step_completed(prev_state, prev_action, reward, state, done)
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
        scores = []
        for _ in range(episodes):
            episode = self.run_episode(render=render)
            rewards = total_rewards(episode)
            scores.append(rewards)
            print('score=',rewards)
        print('Avg rewards=',np.mean(scores))

def total_rewards(episode):
    return sum([reward for (state, action, reward, next_state, done) in episode])