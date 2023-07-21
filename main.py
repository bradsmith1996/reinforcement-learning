# Author: Brad Smith
# Title: Main Program for Lunar Lander OpenAi Gym Problem
# Class: CS 7642 - Reinforcement Learning
# Summer 2021
import gym
from gym.envs import box2d
from gym.envs.box2d import lunar_lander
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import random
from collections import deque, namedtuple
# Local Includes:
from data_logger import log_lunar_lander

# Globals:
# This helps with debugging, provides more sig figs
torch.set_printoptions(precision=10)

# Memory: Creates a simple structure type format that is efficicent for
#         passing data into the loss function. Named tuples for replay memory
#         are stored in the class MemoryCapacity
Memory = namedtuple('Memory',
                        ('state', 'action', 'reward', 'state_prime','terminal'))

# MemoryCapacity: Class responsible for replay memory.
#                 - Stores memories, provides random batches of memories to learner
class MemoryCapacity:
   def __init__(self,capacity_count):
      # Initialize deque with provided capacity, initialize to empty:
      self.capacity = capacity_count
      self.memory_bank = deque([],maxlen=capacity_count)
   def get_batch(self,batch_size):
      # Pulls random number of memories currently being held
      return random.sample(self.memory_bank,batch_size)
   def push(self,*args):
      # Pushes into the buffer:
      self.memory_bank.append(Memory(*args))
   def size(self):
      return len(self.memory_bank)
   def get_capacity(self):
      return self.capacity

# How to Build a Feedforward Neural Network in torch: https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb
# Limited to two layers, no reason to get complicated with this
class FeedForwardNeuralNetwork(torch.nn.Module):
   def __init__(self,input_size, output_size, hidden_layer_size):
      # Initialize super class
      super(FeedForwardNeuralNetwork, self).__init__()
      # Create the layers
      self.fc1 = torch.nn.Linear(input_size, hidden_layer_size)
      self.relu1 = torch.nn.ReLU()
      self.fc2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
      self.relu2 = torch.nn.ReLU()
      self.fc3 = torch.nn.Linear(hidden_layer_size, output_size)
   def forward(self,x):
      hidden1 = self.fc1(x)
      relu1_ = self.relu1(hidden1)
      hidden2 = self.fc2(relu1_)
      relu2_ = self.relu2(hidden2)
      output = self.fc3(relu2_)
      return output

class LunarLanderAgent:
   # Constructor:
   def __init__(self, render_mode="rgb_array"):
      # Set up the Environment and its parameters:
      self.num_actions = 4 # Number of lander actions: do nothing, thrust left, thrust right, main engine
      self.env = gym.make("LunarLander-v2", render_mode=render_mode)
      self.env.reset() # Reset the lander to begin
      # Meta Data Collection:
      self.total_reward_vector = []
      self.epsilon_schedule = []
      self.network_loss = []
      # Hyper Parameters:
      self.num_episodes = 600
      # Hyper Parameters:
      self.gamma = 0.99
      self.memory_capacity_count = 10000 # tuning knob for eventual deep Q -learning, this is about 9-15 episodes of memory
      self.target_update_rate = 5000 # How many steps through the agent before updating the target network
      self.batch_size = 128 # tuning knob for eventual deep Q -learning
      self.epsilson_begin = 0.99 # Common between epsilon decay schedules, defines initial epsilon
      self.epsilson_end = 0.01   # For 'EXPONENTIAL' epsilon decay, defines the limit to infinity, for linear, lower limit of decay
      self.episodes_decay = 50   # Exponential decay rate. How intensely to decay epsilon
      self.linear_decay_rate = 0.99 # For 'LINEAR' epsilon decay, defines the rate
      self.epsilson_schedule_type = 'LINEAR' # OPTIONS: 'EXPONENTIAL' or 'LINEAR'
      self.h = 128
      self.alpha = 0.001
      # Initialize the neural network for value function estimation:
      self.net = FeedForwardNeuralNetwork(8,4,self.h)
      self.target_net = FeedForwardNeuralNetwork(8,4,self.h)
      self.target_net.load_state_dict(self.net.state_dict())
      self.target_net.eval()
      # Initialize Static Parameters
      self.episode_step = 0
      self.epsilon = 0.0
      self.seed = 0
      # Initialize memory:
      self.memory_capacity = MemoryCapacity(self.memory_capacity_count)
   # Wrapper Environment Calls to LunarLander:
   def reset(self):
      return self.env.reset()
   def render(self):
      return self.render()
   def step(self,action):
      return self.env.step(action)
   def set_seed(self,seed):
      self.seed = seed
   def seed_env(self):
      self.env.seed(self.seed)
   # Action Selection Method:
   def select_action(self, net, state, epsilon):
      # Compute decayed epsilon:
      # Source : https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
      # Get the probability seed to take the greedy action:
      if np.random.random(1) > epsilon:
         actions = net.forward(torch.tensor(state))
         action = np.argmax(actions.detach().numpy())
      else:
         # Pick random action otherwise:
         action = np.random.randint(self.num_actions)
      return action
   def train(self,render=False):
      # Train the lander:
      criterion = torch.nn.MSELoss()
      optimizer = torch.optim.SGD(self.net.parameters(), lr=self.alpha)
      self.epsilon = self.epsilson_begin
      counts = 0
      for i in range(self.num_episodes):
         print("Current interation and epsilon: {0} / {1}".format(i,self.epsilon))
         state, _ = np.array(self.reset())
         total_reward = 0
         while True:
            # Select action (max at probability of epsilon (epsilon-greedy))
            action = self.select_action(self.net, state, self.epsilon)
            # Execute action:
            state_prime, reward, done, _, _ = self.step(action)
            total_reward+=reward # Increment total accumulated reward for the current episode
            # Memory Bank Update:
            self.memory_capacity.push(state, action, reward, state_prime, done)
            # Train the Neural Network on the random batch:
            if self.memory_capacity.size() >= self.batch_size:
               # Draw a random batch of memories for neural network training
               random_batch = self.memory_capacity.get_batch(self.batch_size)
               batch = Memory(*zip(*random_batch))
               # Create pytorch batches:
               state_batch = torch.tensor(tuple(batch.state))
               action_batch = torch.tensor(tuple(batch.action))
               reward_batch = torch.tensor(tuple(batch.reward))
               state_prime_batch = torch.tensor(tuple(batch.state_prime))
               terminal_batch = torch.tensor(tuple(batch.terminal))
               # Get batch network outputs:
               output = self.net.forward(state_batch)
               output_prime = self.target_net(state_prime_batch)
               # Construct Target:
               target = reward_batch + (1.0 - terminal_batch.float())*self.gamma*torch.max(output_prime,1)[0]
               # Construct Q(s,a):
               Qsa = output.gather(1, action_batch.unsqueeze(1)).squeeze(1)
               optimizer.zero_grad()
               loss = criterion(Qsa,target.float())
               self.network_loss.append(loss.detach().numpy())
               loss.backward()
               optimizer.step()
            # Every certain number of iterations, update the target network:
            if render:
               self.env.render()
            if done:
               print(total_reward)
               self.epsilon_schedule.append(self.epsilon)
               self.total_reward_vector.append(total_reward)
               self.episode_step+=1
               if self.epsilson_schedule_type == 'EXPONENTIAL':
                  self.epsilon = self.epsilson_end + (self.epsilson_begin-self.epsilson_end) * math.exp(-1.0 * self.episode_step / self.episodes_decay)
               elif self.epsilson_schedule_type == 'LINEAR':
                  self.epsilon*=self.linear_decay_rate
                  if self.epsilon < self.epsilson_end:
                     self.epsilon = self.epsilson_end
               break
            if counts % self.target_update_rate == 0:
               self.target_net.load_state_dict(self.net.state_dict())
            counts+=1
            state = state_prime.copy()
      # Finished, log the data:
      log_lunar_lander(self,"studies/test.txt")
      # Once done running, save the network for persistence:
      model_path = 'network.pth'
      torch.save(self.net,model_path)
      # TO LOAD: net = torch.load(MODEL_PATH)
   def run(self,a_model_path,n_episodes,render=True):
      # Run the lander:
      # self.seed_env()
      net = torch.load(a_model_path)
      net.eval()
      self.epsilon = -1.0 # Always takes greedy choice in select action
      for i in range(n_episodes):
         print("Current interation: {0}".format(i))
         state, _ = np.array(self.reset())
         total_reward = 0
         while True:
            # Select action (max at probability of epsilon (epsilon-greedy))
            action = self.select_action(net,state,self.epsilon)
            # Execute action:
            state_prime, reward, done, _, _ = self.step(action)
            total_reward+=reward # Increment total accumulated reward for the current episode
            # Every certain number of iterations, update the target network:
            if render:
               self.env.render()
            if done:
               print(total_reward)
               self.total_reward_vector.append(total_reward)
               break
            state = state_prime.copy()
      self.seed+=1

if __name__ == '__main__':
   train = True
   model_name = 'network.pth' # NOTE: Must be changed if don't want to overwrite previous
   run = True
   n_episodes = 100
   run_model_name = 'network.pth'
   seed = 1200
   # Train:
   if train:
      lander = LunarLanderAgent()
      lander.train(render=True)
      plt.plot(lander.total_reward_vector,'k')
      plt.xlabel("Episode Number")
      plt.ylabel("Accumulated Reward")
      plt.show()
      plt.plot(lander.epsilon_schedule,'k')
      plt.xlabel("Episode Number")
      plt.ylabel("Epsilon (Probability of taking random action)")
      plt.show()
      plt.plot(lander.network_loss,'k')
      plt.xlabel("Episode Number")
      plt.ylabel("Feedforward Neural Network Loss")
      plt.show()
   if run:
      lander = LunarLanderAgent(render_mode="human")
      # lander.reset(seed=seed)
      lander.run(run_model_name,n_episodes,render=True)
      average = np.mean(lander.total_reward_vector)
      print("Average Reward: {0}".format(average))
      average_array = np.ones([len(lander.total_reward_vector)])*average
      target_average_array = np.ones([len(lander.total_reward_vector)])*200
      fig = plt.figure()
      ax = fig.add_subplot(111)
      plt.plot(lander.total_reward_vector,'ko',label="Accumulated Reward")
      plt.plot(average_array,'b',label="Mean Over All Episodes ("+str(round(average,2))+")")
      plt.plot(target_average_array,'r',label="Target Mean for 100 Episodes")
      plt.xlabel("Episode Number",fontsize=14)
      plt.ylabel("Accumulated Reward",fontsize=14)
      plt.legend()
      plt.show()