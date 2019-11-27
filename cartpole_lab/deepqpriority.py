from deeprico import QLearningPolicy
from probabilitybag import 

class DeepQPrioritized(QLearningPolicy):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.snapshots = None # We'll use the probability bag instead
        self.experience 
    
    #def suggest_action() is same

    def step_completed(self, prev_state, prev_action, reward, state, done):
        self._decay_epsilon()

