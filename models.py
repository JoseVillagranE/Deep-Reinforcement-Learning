import torch
import torch.nn as nn
import torch.nn.functional as F

# ref: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
class Critic(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        """
            initial_layer (dict)
            end_layer (dict)

        """
        super().__init__()

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.linear_3(x)
        return x

class Actor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super().__init__()

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = F.relu(self.linear_1(state))
        x = F.relu(self.linear_2(x))
        x = torch.tanh(self.linear_3(x))
        return x
