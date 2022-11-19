import torch

from torch import nn


class Gan(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('ones', torch.tensor(1.0))
        self.register_buffer('zeros', torch.tensor(0.0))
        self.loss = nn.BCEWithLogitsLoss()

    def get_tensor(self, ones, predicates):
        if ones:
            tensor = self.ones
        else:
            tensor = self.zeros
        return tensor.expand_as(predicates)

    def __call__(self, ones, predicates):
        tensor = self.get_tensor(ones, predicates)
        return self.loss(predicates, tensor)
