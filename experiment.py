import gym
import cartpole_lab.qhindsight

env = gym.make('CartPole-v1')
qhindsight = cartpole_lab.qhindsight.q_hindsight_agent(env)
qhindsight.train(1000000)