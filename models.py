import torch

from torch import nn


# Generator
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # First block coding
        self.b1_conv2d = nn.Conv2d(1, 25, kernel_size=4, stride=2, padding=1, bias=False)

        # Second block coding
        self.b2_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.b2_conv2d = nn.Conv2d(25, 50, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.b2_batch_norm2d_1 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Third block coding
        self.b3_leaky_relu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.b3_conv2d = nn.Conv2d(50, 100, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.b3_batch_norm2d_1 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Fourth block coding
        self.b4_leaky_relu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.b4_conv2d = nn.Conv2d(100, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.b4_batch_norm2d_1 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Middle block
        self.mb_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mb_conv2d = nn.Conv2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb_relu = nn.ReLU(inplace=True)
        self.mb_conv_transpose2d = nn.ConvTranspose2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb_batch_norm2d = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Fourth block decoding
        self.b4_relu = nn.ReLU(inplace=True)
        self.b4_conv_transpose2d = nn.ConvTranspose2d(400, 100, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.b4_batch_norm2d_2 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Third block decoding
        self.b3_relu = nn.ReLU(inplace=True)
        self.b3_conv_transpose2d = nn.ConvTranspose2d(200, 50, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.b3_batch_norm2d_2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Second block decoding
        self.b2_relu = nn.ReLU(inplace=True)
        self.b2_conv_transpose2d = nn.ConvTranspose2d(100, 25, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.b2_batch_norm2d_2 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Third block decoding
        self.b1_relu = nn.ReLU(inplace=True)
        self.b1_conv_transpose2d = nn.ConvTranspose2d(50, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.b1_batch_norm2d = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        def init_weights(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Init weights
        self.apply(init_weights)
    
    def forward(self, x):
        # First block coding
        x = self.b1_conv2d(x)
        b1_connection = x

        # Second block coding
        x = self.b2_leaky_relu(x)
        x = self.b2_conv2d(x)
        x = self.b2_batch_norm2d_1(x)
        b2_connection = x

        # Third block coding
        x = self.b3_leaky_relu(x)
        x = self.b3_conv2d(x)
        x = self.b3_batch_norm2d_1(x)
        b3_connection = x

        # Fourth block coding
        x = self.b4_leaky_relu(x)
        x = self.b4_conv2d(x)
        x = self.b4_batch_norm2d_1(x)
        b4_connection = x

        # Middle block
        x = self.mb_leaky_relu(x)
        x = self.mb_conv2d(x)
        x = self.mb_relu(x)
        x = self.mb_conv_transpose2d(x)
        x = self.mb_batch_norm2d(x)

        # Fourth block decoding
        x = self.b4_relu(torch.cat([b4_connection, x], 1))
        x = self.b4_conv_transpose2d(x)
        x = self.b4_batch_norm2d_2(x)

        # Third block decoding
        x = self.b3_relu(torch.cat([b3_connection, x], 1))
        x = self.b3_conv_transpose2d(x)
        x = self.b3_batch_norm2d_2(x)

        # Second block decoding
        x = self.b2_relu(torch.cat([b2_connection, x], 1))
        x = self.b2_conv_transpose2d(x)
        x = self.b2_batch_norm2d_2(x)

        # First block decoding
        x = self.b1_relu(torch.cat([b1_connection, x], 1))
        x = self.b1_conv_transpose2d(x)
        x = self.b1_batch_norm2d(x)

        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=True),
          nn.LeakyReLU(0.2, True),
          nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2, True),
          nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(0.2, True),
          nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(512),
          nn.LeakyReLU(0.2, True),
          nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True)
        )
        
        # Initialize weights
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                

    def forward(self, x):
        return self.model(x)