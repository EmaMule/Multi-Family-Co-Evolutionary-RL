from torch import nn
import torch


# a simple neural network adapted to the three explored environments
class ClassicNet(nn.Module):

    def __init__(self, input_shape = [3,3,2], n_actions = 9, bias = True):
        super(ClassicNet, self).__init__()
        if len(input_shape) == 1: # observation space is a vector
            self.fc1 = nn.Linear(input_shape[0], 256, bias=bias)
        else:
            self.fc1 = nn.Linear(2 * input_shape[0] * input_shape[1], 256, bias = bias)

        self.fc2 = nn.Linear(256, 32, bias=bias)
        self.fc3 = nn.Linear(32, n_actions, bias=True) #always false

        # trying with 64 and 32 instead of 128 and 64


    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))  # Using tanh activation function
        x = torch.relu(self.fc2(x))  # Using tanh activation function
        x = self.fc3(x)
        return x
    

# a simple deeper network
class DeepNet(nn.Module):

    def __init__(self, input_shape = [3,3,2], n_actions = 9, bias = True):
        super(DeepNet, self).__init__()

        if len(input_shape) == 1: #observation space is a vector
          self.fc1 = nn.Linear(input_shape[0], 64, bias = bias)
        else:
          self.fc1 = nn.Linear(2 * input_shape[0]*input_shape[1], 64, bias = bias)

        self.fc2 = nn.Linear(64, 32, bias = bias)
        self.fc3 = nn.Linear(32, 32, bias = bias)
        self.fc4 = nn.Linear(32, 16, bias = bias)
        self.fc5 = nn.Linear(16, 16, bias = bias)
        self.fc6 = nn.Linear(16, n_actions, bias = True)


    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x