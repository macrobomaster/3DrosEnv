import torch
import torch.nn as nn

class CriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CriticRNN, self).__init__()
        self.rnn = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        print(x[:, :10])
        # print(torch.sum(x))
        # print("Critic X shape", x.shape)
        output = self.rnn(x)
        # Take the last output of the sequence
        # print("Critic output shape: ", output.shape)
        # print(output)
        output = output[:, :]  # Shape: (batch_size, hidden_size)
        # print("Critic input shape: ", output.shape)
        output = self.fc1(output)
        # print("Critic output shape: ", output.shape)
        return output


