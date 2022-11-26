import torch

from torch import nn, optim

from Utilities.Submodules.discriminator import Discriminator
from Utilities.Submodules.unet import Unet
from Utilities.Submodules.ganloss import Gan


class Network(nn.Module):
    def __init__(self, device, resnet=None):
        super().__init__()
        
        self.device = device
        self.discriminator = self.make_model(Discriminator())

        if resnet is not None:
            self.unet = self.make_model(Unet())
        else:
            self.unet = resnet

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

    def forward(self):
        self.gen_colours = self.unet(self.L)

    def set_required_grads(self, model, value):
        for parameter in model.parameters():
            parameter.requires_grad = value

    def backward_1(self):
        gen_img = torch.cat([self.L, self.gen_colours], dim=1)
        gen_predicates = self.discriminator(gen_img.detach())
        self.gen_gan_loss = self.gan(False, gen_predicates)
        train_img = torch.cat([self.L, self.ab], dim=1)
        train_predicates = self.discriminator(train_img)
        self.train_gan_loss = self.gan(True, train_predicates)
        self.gan_loss = (self.gen_gan_loss + self.train_gan_loss) * 0.5
        self.gan_loss.backward()
    
    def backward_2(self):
        gen_img = torch.cat([self.L, self.gen_colours], dim=1)
        gen_predicates = self.discriminator(gen_img)
        self.gan_loss = self.gan(True, gen_predicates)
        self.loss_loss = self.loss(self.gen_colours, self.ab) * 100.
        self.loss_combined = self.gan_loss + self.loss_loss
        self.loss_combined.backward()

    def optimize(self):
        self.forward()
        self.discriminator.train()
        self.set_required_grads(self.discriminator, True)
        self.opt_2.zero_grad()
        self.backward_1()
        self.opt_2.step()\
        
        self.unet.train()
        self.set_required_grads(self.discriminator, False)
        self.opt_1.zero_grad()
        self.backward_2()
        self.opt_1.step()
