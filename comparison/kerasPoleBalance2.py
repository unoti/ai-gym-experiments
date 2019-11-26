import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from PIL import Image

from loggingFunctionality import LoggerOutputs # Import ScoreLogger Class from /scores/score_logger

# Define which openAI Gym environment to use
ENV_NAME = "CartPole-v1"

# User options
DEBUG = False
LOAD_PRIOR_MODEL = False
PRIOR_MODEL_NAME = "kerasModelWeights.h5"
EXPORT_MODEL = False
SAVE_GIFS = False

# Define RL variables
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 10

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DeepQBenPolicy(LoggerOutputs):
    def __init__(self, ENV_NAME, env):
        # init inherited class
        super().__init__(ENV_NAME)

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
        
        epsilon_orig = self.policy.epsilon # Save the original epsilon, in case we want to do external things with agent later.
        self.policy.epsilon = epsilon_start
        
        run = 0
        while True:
            run += 1
            episode = self.run_episode(render=render)
            total_rewards = len(episode)
            episode_scores.append(total_rewards)
            epsilons.append(self.policy.epsilon)
            self.policy.episode_completed(episode)

            print ("Run: " + str(run) + ", exploration: " + str(self.policy.exploration_rate) + ", score: " + str(total_rewards))
            self.policy.addScore(total_rewards, run) # Call inherited method              

            #now = time.time()
            #if now - last_update_time > image_update_seconds:
            #    plot_episode_scores(episode_scores, epsilons, save=chartfile)
            #    last_update_time = now

        self.policy.epsilon = epsilon_orig # Restore epsilon.


'''
Main method for solving the pole balance cart problem from OpenAI gym.
Initiates multiple runs containing multiple iterations to solve.
Between runs it resets the gym environment.
Results are logged and dependent on user settings creates GIFs of render
and exports/saves Keras model.
'''
def poleBalance2():
    env = gym.make(ENV_NAME) # New OpenAI Gym Env
    observation_space = env.observation_space.shape[0] # four observations cart pos&vel & pole angle&vel
    action_space = env.action_space.n # two actions cart goes left or right
    
    # DeepQLearningSolver class object with observation and action space attributes, and inherited class attribute
    #dqnObj = DeepQLearningSolver(ENV_NAME, observation_space, action_space)
    dqnObj = DeepQBenPolicy(ENV_NAME, env)
       
    frames = [] # Empty list to be filled with frames for GIF export
    run = 0 # Start run calculations from zero
    while True: # Each actual simulation run instance
        if DEBUG:
            print("Starting New Simulation...")
        run += 1 # Started a new simulation run so add one
        state = env.reset() # Reset Gym environment
        #state = np.reshape(state, [1, observation_space]) # random new state for this run instance
        step = 0 # init starting simulation step at zero
        while True: # Each timestep within said simulation...
            if DEBUG:
                print("Starting New Time Step...")
            step += 1 # New timestep
            #env.render() # Render image of current gym simulation
            
            # Call object method act which delivers a 0 or 1 based upon exploration rate/decay (1 == Right & 0 == Left)
            #action = dqnObj.act(state) # state will be a first iteration new state, or will be the last time steps state
            action = dqnObj.suggest_action(state) # state will be a first iteration new state, or will be the last time steps state
            if DEBUG:
                if action == 0:
                    print("Cart push Left...")
                elif action == 1:
                    print("Cart push Right...")
            
            '''
            time step forward using random action but as exploration decays use 0 less and 1 more.
            .step creates an Observation(state_next = object), reward(float), terminal(done = bool), info(dict) for each time step
            state_next is essentially what is going on in the gym, rotations velocities etc.
            '''
            # Using action against the current state what has happened - step forward to find out...          
            state_next, reward, terminal, info = env.step(action) 
           
            # If the simulation has not terminated (i.e. failed criteria) then it gets a positive reward.
            # If it has terminated, i.e. the pole has fallen over/fail criteria met then it gets a negative reward
            reward = reward if not terminal else -reward
                        
            state = np.reshape(state, [1, observation_space]) # random new state for this run instance
            state_next = np.reshape(state_next, [1, observation_space])
            
            # Activly rememeber what state you were in, what action you took, 
            # whether that was "rewarding" and what the next state was and then whether it terminated or not.
            dqnObj.remember(state, action, reward, state_next, terminal)
            
            state = state_next # Define state as that of your prior attempt - i.e. previous step influences this step i.e. learninggggggg
            
            #frames.append(Image.fromarray(env.render(mode='rgb_array')))  # save each frames

            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqnObj.exploration_rate) + ", score: " + str(step))
                dqnObj.addScore(step, run) # Call inherited method              
                if run % 5 == 0: # Export every 5th run                       
                    if EXPORT_MODEL:
                        if DEBUG:
                            print("Exporting Model")
                        dqnObj.exportModel()
                    if run % 20 == 0: # Export GIF of latest 5 runs every 20 runs
                        gifPath = "./GIFs/cart_R" + str(run) + ".gif"
                        if DEBUG:
                            print("Creating GIF: " + gifPath)  
                        if SAVE_GIFS:
                            dqnObj.exportGIF(gifPath, frames)      
                    frames = [] # Reset

                break            
            dqnObj.experienceReplay() # Actual reinforcement...

def test_agent():
    env = gym.make(ENV_NAME) # New OpenAI Gym Env
    agent = Agent(env, DeepQBenPolicy(ENV_NAME, env))
    agent.train(episodes=500)

if __name__ == "__main__":
    poleBalance2() # call poleBalance() method
    #test_agent()