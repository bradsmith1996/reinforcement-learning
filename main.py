import gym
from gym.envs import box2d
from gym.envs.box2d import lunar_lander
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import math
import random
from collections import deque, namedtuple
# Local Includes:
from data_logger import log_lunar_lander
from typing import List
from tqdm import tqdm

# Memory: Creates a simple structure type format that is efficicent for
#    passing data into the loss function. Named tuples for replay memory
#    are stored in the class MemoryCapacity.
Memory = namedtuple('Memory',
                        ('state', 'action', 'reward', 'state_prime','terminal'))

# MemoryCapacity: Class responsible for replay memory. Stores memories,
# provides random batches of memories to learner.
class MemoryCapacity:
   def __init__(self,capacity_count):
      # Initialize deque with provided capacity, initialize to empty.
      self.capacity = capacity_count
      self.memory_bank = deque([],maxlen=capacity_count)
   def get_batch(self,batch_size):
      # Pulls random number of memories currently being held.
      return random.sample(self.memory_bank,batch_size)
   def push(self,*args):
      # Pushes experience into the buffer.
      self.memory_bank.append(Memory(*args))
   def size(self):
      return len(self.memory_bank)
   def get_capacity(self):
      return self.capacity

# Neural Network.
class FeedForwardNeuralNetwork(torch.nn.Module):
   # Activation dictionary.
   activations = {
      "relu": nn.ReLU(),
      "tanh": nn.Tanh(),
   }
   def __init__(
      self,
      input_size: int,
      output_size: int,
      hidden_layer_sizes: List[int],
      activations = List[str]
      ):
      # Initialize the super class.
      super(FeedForwardNeuralNetwork, self).__init__()
      # Initialize the layers with input layer only.
      self.layers = nn.Sequential(
         nn.Linear(input_size, hidden_layer_sizes[0]),
         self.activations[activations[0]],
      )
      # Create hidden layers.
      for idx in range(len(hidden_layer_sizes)-1):
         # Add in linear layer.
         self.layers.append(
            nn.Linear(hidden_layer_sizes[idx], hidden_layer_sizes[idx+1]),
         )
         # Add in non-linear activation after linear layer.
         self.layers.append(
            self.activations[activations[idx+1]]
         )
      # Create the output layer.
      self.layers.append(
         nn.Linear(hidden_layer_sizes[-1], output_size)
      )
   def forward(self,x):
      return self.layers(x)

class DeepQLearningAgent:
   def __init__(
      self,
      environment: str,
      render: bool = False,
      progress_bar: bool = False,
   ) -> None:
      # Set up the Environment and its parameters.
      self.progress_bar = progress_bar
      self.reward_history_sma = deque([], maxlen=50)
      render_mode = "human" if render else "rgb_array"

      # Load the environment and get environment specific information.
      self.env = gym.make(environment, render_mode=render_mode)
      self.num_actions = self.env.action_space.n
      self.num_states = self.env.observation_space.shape[0]

      # Reset the lander to begin.
      self.env.reset()
      # Meta Data Collection:
      self.total_reward_vector = []
      self.epsilon_schedule = []
      self.network_loss = []
      # Hyper Parameters.
      self.gamma = 0.99
      self.memory_capacity_count = 10000 # tuning knob for eventual deep Q -learning, this is about 9-15 episodes of memory
      self.target_update_rate = 5000 # How many steps through the agent before updating the target network
      self.batch_size = 128 # tuning knob for eventual deep Q -learning
      self.epsilson_begin = 0.99 # Common between epsilon decay schedules, defines initial epsilon
      self.epsilson_end = 0.01   # For 'EXPONENTIAL' epsilon decay, defines the limit to infinity, for linear, lower limit of decay
      self.episodes_decay = 50   # Exponential decay rate. How intensely to decay epsilon
      self.linear_decay_rate = 0.99 # For 'LINEAR' epsilon decay, defines the rate
      self.epsilson_schedule_type = 'LINEAR' # OPTIONS: 'EXPONENTIAL' or 'LINEAR'
      self.hidden_sizes = [128, 128]
      activations = ["relu", "relu"]
      self.alpha = 0.001
      self.episode_step = 0
      # Initialize the neural network for value function estimation.
      self.net = FeedForwardNeuralNetwork(self.num_states,self.num_actions,self.hidden_sizes, activations)
      self.target_net = FeedForwardNeuralNetwork(self.num_states,self.num_actions,self.hidden_sizes, activations)
      self.target_net.load_state_dict(self.net.state_dict())
      self.target_net.eval()
      # Initialize memory.
      self.memory_capacity = MemoryCapacity(self.memory_capacity_count)
      # Initialize reward:
      self.minimum_reward = -300
   # Wrapper Environment Calls to LunarLander.
   def reset(self):
      return self.env.reset()
   def render(self):
      return self.render()
   def step(self,action):
      return self.env.step(action)
   # Action Selection Method.
   def select_action(self, net, state, epsilon):
      # Compute decayed epsilon.
      # Get the probability seed to take the greedy action.
      if np.random.random(1) > epsilon:
         actions = net.forward(torch.tensor(state))
         action = np.argmax(actions.detach().numpy())
      else:
         # Pick random action otherwise.
         action = np.random.randint(self.num_actions)
      return action
   def train(self, num_episodes: int = 600, render: bool = False):
      # Train the lander.
      reward_history = open("reward.txt", "a")
      criterion = torch.nn.MSELoss()
      optimizer = torch.optim.SGD(self.net.parameters(), lr=self.alpha)
      epsilon = self.epsilson_begin
      counts = 0
      if self.progress_bar:
         pbar = tqdm(desc="Episode Progress", total=num_episodes, position=0)
         reward_log = tqdm(total=0, position=1, bar_format='{desc}')
         epsilon_log = tqdm(total=0, position=3, bar_format='{desc}')
      for episode_iteration in range(num_episodes):
         state, _ = self.reset()
         total_reward = 0
         episode_terminated = False
         while (not episode_terminated) and (total_reward > self.minimum_reward):
            # Select action (max at probability of epsilon (epsilon-greedy)).
            action = self.select_action(self.net, state, epsilon)
            # Execute action.
            state_prime, reward, episode_terminated, _, _ = self.step(action)
            total_reward+=reward # Increment total accumulated reward for the current episode.
            # Memory Bank Update.
            self.memory_capacity.push(state, action, reward, state_prime, episode_terminated)
            # Train the Neural Network on the random batch.
            if self.memory_capacity.size() >= self.batch_size:
               # Draw a random batch of memories for neural network training.
               random_batch = self.memory_capacity.get_batch(self.batch_size)
               batch = Memory(*zip(*random_batch))
               # Create pytorch batches.
               state_batch = torch.tensor(np.array(batch.state))
               action_batch = torch.tensor(np.array(batch.action))
               reward_batch = torch.tensor(np.array(batch.reward))
               state_prime_batch = torch.tensor(np.array(batch.state_prime))
               terminal_batch = torch.tensor(np.array(batch.terminal))
               # Get batch network outputs.
               output = self.net.forward(state_batch)
               output_prime = self.target_net(state_prime_batch)
               # Construct Target.
               target = reward_batch + (1.0 - terminal_batch.float())*self.gamma*torch.max(output_prime,1)[0]
               # Construct Q(s,a).
               Qsa = output.gather(1, action_batch.unsqueeze(1)).squeeze(1)
               optimizer.zero_grad()
               loss = criterion(Qsa,target.float())
               self.network_loss.append(loss.detach().numpy())
               loss.backward()
               optimizer.step()

            # If flag set to render, render.
            if render:
               self.env.render()

            # Update the target network according to the update rate.
            if counts % self.target_update_rate == 0:
               self.target_net.load_state_dict(self.net.state_dict())
            counts+=1
            state = state_prime.copy()

         # Post-episode: Save off data.
         self.epsilon_schedule.append(epsilon)
         self.total_reward_vector.append(total_reward)
         self.reward_history_sma.append(total_reward)

         # Post-episode: Update epsilon.
         if self.epsilson_schedule_type == 'EXPONENTIAL':
            epsilon = self.epsilson_end + (self.epsilson_begin-self.epsilson_end) * math.exp(-1.0 * episode_iteration / self.episodes_decay)
         elif self.epsilson_schedule_type == 'LINEAR':
            epsilon*=self.linear_decay_rate
            if epsilon < self.epsilson_end:
               epsilon = self.epsilson_end

         # Post-episode: Update the progress bar.
         if self.progress_bar:
            sma = np.mean(self.reward_history_sma)
            reward_log.set_description_str(f'Cumulative Reward SMA (50 episodes): {round(sma, 2)}')
            epsilon_log.set_description_str(f'Current Epsilon: {round(epsilon, 2)}')
            reward_history.write(str(self.reward_history_sma))
            reward_history.write("\n")
            pbar.update(1)

      # Finished, log the data.
      log_lunar_lander(self,"studies/test.txt", num_episodes)
      # Once done running, save the network for persistence.
      model_path = 'network.pth'
      torch.save(self.net,model_path)

   def run(self,a_model_path,num_episodes,render=True):
      # Run the lander.
      net = torch.load(a_model_path)
      net.eval()
      epsilon = -1.0 # Always takes greedy choice in select action.
      if self.progress_bar:
         pbar = tqdm(desc="Episode Progress", total=num_episodes, position=0)
         reward_log = tqdm(total=0, position=1, bar_format='{desc}')
      for i in range(num_episodes):
         state, _ = self.reset()
         total_reward = 0
         episode_terminated = False
         while (not episode_terminated) and (total_reward > self.minimum_reward):
            # Select action (max at probability of epsilon (epsilon-greedy)).
            action = self.select_action(net, state, epsilon)
            # Execute action:
            state_prime, reward, episode_terminated, _, _ = self.step(action)
            total_reward+=reward # Increment total accumulated reward for the current episode.
            # Every certain number of iterations, update the target network.
            if render:
               self.env.render()
            state = state_prime.copy()
         self.total_reward_vector.append(total_reward)
         if self.progress_bar:
            reward_log.set_description_str(f'Episode Cumulative Reward: {round(total_reward, 2)}')
            pbar.update(1)

if __name__ == '__main__':
   train = True
   model_name = 'network.pth'
   test = True
   num_episodes_train = 600
   num_episodes_test = 200
   run_model_name = 'network.pth'
   # Train:
   if train:
      lander = DeepQLearningAgent(environment="LunarLander-v2", render=False, progress_bar=True)
      lander.train(num_episodes=num_episodes_train, render=False)
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
   if test:
      lander = DeepQLearningAgent(environment="LunarLander-v2", render=False, progress_bar=True)
      lander.run(run_model_name,num_episodes_test,render=True)
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