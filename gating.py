from typing import Callable, List
from enum import Enum

import torch
from torch import nn

class Gating(Enum):
    HARD = False
    SOFT = True

    def gate(self)->Callable:
        return [][int(self.value)]


class GatingNetwork(nn.Module):
    def __init__(self, in_dim:int,  num_experts: int, hidden_layer_sizes: int | List[int] = 32, act=nn.LeakyReLU):
        super().__init__()
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = [hidden_layer_sizes]

        layers = [nn.Linear(in_dim, hidden_layer_sizes[0]), act()] 

        for in_dim ,out_dim in zip(hidden_layer_sizes[:-1], hidden_layer_sizes[1:]):
            layers.extend(
                [
                nn.Linear(in_dim, out_dim),
                act()
                    ]
                ) 

        layers.append(nn.Linear(hidden_layer_sizes[-1], num_experts)) 
        self.fc = nn.Sequential(*layers)

    def forward(self, metadata):
        # Shape: (batch_size, num_experts)
        logits = self.fc(metadata)
        # Shape: (batch_size, num_experts)
        expert_weights = torch.softmax(logits, dim=1)
        return expert_weights