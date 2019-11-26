# Deep Q Learning agent
# For comparison, adapted Ben Harris's solution https://github.com/Ben-C-Harris/Reinforcement-Learning-Pole-Balance/blob/master/kerasPoleBalance.py
# into a policy that works in my own framework for easy comparison.

from collections import deque
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from .agent import Agent

# Define RL variables
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 10

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DeepQBenPolicy:
    def __init__(self, env):
        # Set method attributes
        self.exploration_rate = EXPLORATION_MAX
        self.epsilon = self.exploration_rate # Rico's graphing uses this.

        # Set openAI gym space
        self.action_space = env.action_space.n
        self.observation_space = env.observation_space.shape[0]
        
        # Memory managment for RL
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Build sequential keras tensorflow model (Sequential is a linear stack of layers)
        self.model = Sequential() # Define model type
        
        # Build first layer. Input shape is of openAI gym observation space i.e. 4 inputs
        self.model.add(Dense(24, input_shape=(env.observation_space.shape[0],), activation="relu")) # input shape of four due to four observations
        self.model.add(Dense(24, activation="relu")) 
        self.model.add(Dense(self.action_space, activation="linear")) # output two as only push cart to right or left are required
        
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def suggest_action(self, state, epsilon=None):
        '''
        act method returns 0 or 1.
        
        1-exploration_rate of the time itll be random, else returns 1...
        
        Starts more random but as time goes on it learns more and more from prior results.
        i.e. weighting on learning gets higher as you go longer as opposed to earlier where
        it is just trying to be random and see what happens
        '''
        if np.random.rand() < self.exploration_rate: # X% of the time NOTE: Exploration Rate changes with steps due to exploration decay
            return random.randrange(self.action_space) # randomly returns 0 or 1  (i.e. left/right)
        else:
            state = self._adapt_state(state)
            q_values = self.model.predict(state) # Predict quality value - predicts the confidence of using left or right movement of cart
            return np.argmax(q_values[0])
    
    def _adapt_state(self, state):
        # Transform a state into a single-row array of states.
        # Ben's code handles all states as single-rowed batches like this.
        # this is the same as np.expand_dims(state, axis=0)
        return np.reshape(state, [1, self.observation_space])

    def step_completed(self, prev_state, prev_action, reward, state, terminal):
        prev_state = self._adapt_state(prev_state)
        state = self._adapt_state(state)
    
        # If the simulation has not terminated (i.e. failed criteria) then it gets a positive reward.
        # If it has terminated, i.e. the pole has fallen over/fail criteria met then it gets a negative reward
        reward = reward if not terminal else -reward

        # Activly rememeber what state you were in, what action you took, 
        # whether that was "rewarding" and what the next state was and then whether it terminated or not.
        self.remember(prev_state, prev_action, reward, state, terminal)

        self.experienceReplay() # Actual reinforcement...

    def episode_completed(self, episode):
        pass

    def remember(self, state, action, reward, next_state, done):
        '''
        Remember prior time step variables and what to do next.
        Append them into batch memory which is later used for RL fit/predict.
        '''
        self.memory.append((state, action, reward, next_state, done))
        
    def experienceReplay(self):
        '''
        If enough simulation data currently remembered in self.memory, this method will
        iterate through the batch and using GAMMA (user defined above) will deliver new
        model predict and fit. The exploration rate decay is then applied.
        '''
        if len(self.memory) < BATCH_SIZE: # Only use the previous X amount of runs to influence and train the model on.
            return
        batch = random.sample(self.memory, BATCH_SIZE) # Define batch learning size
        for state, action, reward, state_next, terminal in batch:
            q_update = reward # Updated quality score
            if not terminal: # i.e. reward not negative
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0])) # Gamma = discount factor & max predicted quality score for next step
            q_values = self.model.predict(state) # q_values are the confidence/quality values over whether the cart needs to go left or right in next frame
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY # Apply exploration decay
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate) # Ensure doesnt go below specified minimum value
        self.epsilon = self.exploration_rate # Rico's graphing uses this.

def deep_ben(env):
    return Agent(env, DeepQBenPolicy(env))