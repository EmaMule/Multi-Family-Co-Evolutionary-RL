from torch import nn
import torch

class ClassicNet(nn.Module):

    def __init__(self, input_shape = [3,3,2], n_actions = 9):
        super(ClassicNet, self).__init__()

        if len(input_shape) == 1: #observation space is a vector
          self.fc1 = nn.Linear(input_shape[0], 128)
        else:
          self.fc1 = nn.Linear(2 * input_shape[0]*input_shape[1], 128)

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)


    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepNet(nn.Module):

    def __init__(self, input_shape = [3,3,2], n_actions = 9):
        super(DeepNet, self).__init__()

        if len(input_shape) == 1: #observation space is a vector
          self.fc1 = nn.Linear(input_shape[0], 64)
        else:
          self.fc1 = nn.Linear(2 * input_shape[0]*input_shape[1], 64)

        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 16)
        self.fc6 = nn.Linear(16, n_actions)


    def forward(self, x):
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class Connect4Net(nn.Module):

    def __init__(self, input_shape = [6,7,2], n_actions = 7):
        super(Connect4Net, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=(4,4),
            stride=1,
            padding=0,
        )

        self.fc1 = nn.Linear(192, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x