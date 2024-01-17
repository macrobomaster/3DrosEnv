
import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.tanh(self.fc1(state))
        return self.fc2(x)


