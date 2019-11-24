from .progress import log_progress
from .charts import plot_episode_lengths
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
        episode_lengths = []
        epsilons = []
        image_update_seconds = 5
        last_update_time = time.time()
        
        epsilon_orig = self.policy.epsilon # Save the original epsilon, in case we want to do external things with agent later.
        self.policy.epsilon = epsilon_start
        
        chartfile = r'images/progress.png'
        for i in log_progress(range(episodes), name='Episodes'):
            episode = self.run_episode(render=render)
            episode_lengths.append(len(episode))
            epsilons.append(self.policy.epsilon)
            self.policy.episode_completed(episode)
            now = time.time()
            #print(now, now-last_update_time)
            if now - last_update_time > image_update_seconds:
                plot_episode_lengths(episode_lengths, epsilons, save=chartfile)
                last_update_time = now
            
        plot_episode_lengths(episode_lengths, epsilons, save=chartfile)
        print('Max length=%f avg=%f' % (np.max(episode_lengths), np.mean(episode_lengths)))
        
        self.policy.epsilon = epsilon_orig # Restore epsilon.