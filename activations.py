import torch
from torch import nn

def sin_(input):
    return torch.sin(input) 

class Sin(nn.Module):

    def __init__(self):

        super().__init__() # init the base class

    def forward(self, input):

        return sin_(input)

#comment_