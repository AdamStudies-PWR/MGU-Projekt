import torch

from torch import nn, optim

from Utilities.Submodules.discriminator import Discriminator
from Utilities.Submodules.unet import Unet
from Utilities.Submodules.ganloss import Gan


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Got device: " + str(self.device))

        self.discriminator = self.make_model(Discriminator())
        self.unet = self.make_model(Unet(self.device))
        self.gan = Gan().to(self.device)
        self.loss = nn.L1Loss()
        self.opt_1 = optim.Adam(self.unet.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_2 = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    def set_up_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def set_up_weights(self, model):
        def set_up(model):
            classname = model.__class__.__name__
            if hasattr(model, 'weight') and 'Conv' in classname:
                nn.init.normal_(model.weight.data, mean=0.0, std=0.02)
                if hasattr(model, 'bias') and model.bias is not None:
                    nn.init.constant_(model.bias.data, 0.0)
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(model.weight.data, 1., 0.02)
                nn.init.constant_(model.bias.data, 0.)

        model.apply(set_up)
        return model

    def make_model(self, model):
        model = model.to(self.device)
        model = self.set_up_weights(model)
        return model
