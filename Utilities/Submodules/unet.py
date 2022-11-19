import torch

from torch import nn


class BaseUnet(nn.Module):
    def __init__(self, f1, f2, f3):
        super().__init__ ()
        self.conv = nn.Conv2d(f3, f2, kernel_size=2, stride=1, padding=0, bias=False)
        self.leaky = nn.LeakyReLU(0.2, True)
        self.norm_1 = nn.BatchNorm2d(f2)
        self.relu = nn.ReLU(True)
        self.norm_2 = nn.BatchNorm2d(f1)


class InnerBlock(BaseUnet):
    def __init__(self, f1, f2, f3):
        super().__init__(f1, f2, f3)
        self.model = self.make_model(f1, f2)
    
    def make_model(self, f1, f2):
        trans_conv = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=1, padding=0, bias=False)
        down = [self.leaky, self.conv]
        up = [self.relu, trans_conv, self.norm_2]
        model = down + up
        return nn.Sequential(*model)

    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


class MiddleBlock(BaseUnet):
    def __init__(self, f1, f2, f3, submodule, dropout=True):
        super().__init__(f1, f2, f3)
        self.model = self.make_model(f1, f2, submodule, dropout)

    def make_model(self, f1, f2, submodule, dropout):
        trans_conv = nn.ConvTranspose2d(f2 *  2, f1, kernel_size=2, stride=1, padding=0, bias=False)
        down = [self.leaky, self.conv, self.norm_1]
        up = [self.relu, trans_conv, self.norm_2]
        if dropout:
            up = up + [nn.Dropout(0.5)]
        model = down + [submodule] + up
        return nn.Sequential(*model)
    
    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


class OuterBlock(BaseUnet):
    def __init__(self, f1, f2, f3, submodule):
        super().__init__(f1, f2, f3)
        self.model = self.make_model(f1, f2, submodule)
    
    def make_model(self, f1, f2, submodule):
        trans_conv = nn.ConvTranspose2d(f2 * 2, f1, kernel_size=2, stride=1, padding=0)
        down = [self.conv]
        up = [self.relu, trans_conv, nn.Tanh()]
        model = down + [submodule] + up
        return nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Unet(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.filters = 200
        self.model = self.make_model()

    def make_model(self):
        block = InnerBlock(self.filters, self.filters, self.filters)
        for _ in range(1): # Should be 3
            block = MiddleBlock(self.filters, self.filters, self.filters, submodule=block, dropout=True)
        for _ in range(3):
            block = MiddleBlock(self.filters // 2, self.filters, self.filters // 2, submodule=block, dropout=False)
            self.filters //= 2
        return OuterBlock(2, self.filters, 1, submodule=block)

    def forward(self, x):
        return self.model(x)
