from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = self.make_model()

    def make_model(self):
        model = [self.make_layers(3, 25, 2, norm=True, act=False)]
        model = model + [self.make_layers(25 * 2 ** i, 25 * 2 ** (i + 1), 1 if i == 2 else 2, norm=False, act=True)
            for i in range(3)]
        model = model + [self.make_layers(200, 1, 1, norm=False, act=False)]
        return nn.Sequential(*model)


    def make_layers(self, in_channels, out_channels, stride, norm, act):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=not norm)]
        if norm: layers = layers + [nn.BatchNorm2d(out_channels)]
        if act: layers = layers + [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
