import torch

from torch import nn


class InputBlock(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.model = self.build_net(input, output)
    
    def build_net(self, input, output):
        leaky = nn.LeakyReLU(0.2, True)
        conv_1 = nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1, bias=False)
        relu = nn.ReLU(True)
        conv_2 = nn.ConvTranspose2d(output, input, kernel_size=4, stride=2, padding=1, bias=False)
        norm = nn.BatchNorm2d(input)
        down = [leaky, conv_1]
        up = [relu, conv_2, norm]
        model = down + up
        return nn.Sequential(*model)
    
    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


class MiddleBlock(nn.Module):
    def __init__(self, input, output, submodule, dropout):
        super().__init__()
        self.model = self.build_net(input, output, submodule, dropout)
    
    def build_net(self, input, output, submodule, dropout):
        leaky = nn.LeakyReLU(0.2, True)
        conv_1 = nn.Conv2d(input, output, kernel_size=4, stride=2, padding=1, bias=False)
        norm_1 = nn.BatchNorm2d(output)
        relu = nn.ReLU(True)
        conv_2 = nn.ConvTranspose2d(output * 2, input, kernel_size=4, stride=2, padding=1, bias=False)
        norm_2 = nn.BatchNorm2d(input)
        down = [leaky, conv_1, norm_1]
        up = [relu, conv_2, norm_2]
        if dropout: up = up + [nn.Dropout(0.5)]
        model = down + [submodule] + up
        return nn.Sequential(*model)

    def forward(self, x):
        return torch.cat([x, self.model(x)], 1)


class OutputBlock(nn.Module):
    def __init__(self, input, output, submodule):
        super().__init__()
        self.model = self.build_net(input, output, submodule)
    
    def build_net(self, input, output, submodule):
        conv_1 = nn.Conv2d(2, output, kernel_size=4, stride=2, padding=1, bias=False)
        relu = nn.ReLU(True)
        conv_2 = nn.ConvTranspose2d(output * 2, input, kernel_size=4, stride=2, padding=1)
        norm = nn.BatchNorm2d(input)
        down = [conv_1]
        up = [relu, conv_2, norm]
        model = down + [submodule] + up
        return nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.filters = 200
        self.model = self.build_net()

    def build_net(self):    
        block = InputBlock(self.filters, self.filters)
        for _ in range(3):
            block = MiddleBlock(self.filters, self.filters, block, dropout=True)
        for _ in range(3):
            block = MiddleBlock(self.filters // 2, self.filters, block, dropout=False)
            self.filters //= 2
        return OutputBlock(2, self.filters, block)
    
    def forward(self, x):
        return self.model(x)
