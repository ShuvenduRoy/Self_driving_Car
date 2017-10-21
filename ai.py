# Ai for self driving car

# Importing the libraries
import numpy as np
import random
import os
import torch      # the boss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# Creating the architecture of the neural network
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
# Implementing experience reply
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    def sample(self, batch_size):
        # if list = [(1,2,3), (4,5,6)] then zip(*list) = [(1,4), (2,5), (3,6)]
        samples = zip(*random.sample(self.memory, batch_size))  
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


# Implementing Deep Q Learning
class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters, lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
         probs = F.softmax(self.model(Variable(state, volatile=True))*7)  # temperature parameter
         action = probs.multinomial()
         return action.data[0, 0]
        
    # neural network learning process
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_output = self.model(batch_next_state).detach().max(1)[0]
        target =self.gamma*next_output + batch_reward
        
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variable=True)
        self.optimizer.step()
        
        
        
        
        
