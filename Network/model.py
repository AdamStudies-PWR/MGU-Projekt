from torch import nn

from Network.dyscriminator import Dyscriminator
from Network.generator import Generator


class Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.generator = Generator() # Finish this component
        self.discriminator = Dyscriminator() # Finish this component

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.AB = data['ab'].to(self.device)
    
    def forward(self):
        # self.fakeAB = self.generator(self.L) Generate data based on L (black and white) channel
        pass

    def optimize(self):
        self.forward()
        # do some loss calculations here/optimize generator + discriminator
