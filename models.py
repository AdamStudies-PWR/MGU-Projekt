import torch

from torch import nn

# Generator
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # First block coding
        self.fb_conv2d = nn.Conv2d(1, 25, kernel_size=4, stride=2, padding=1, bias=False)

        # First middle block coding
        self.mb1_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mb1_conv2d = nn.Conv2d(25, 50, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb1_batch_norm2d_1 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Second middle block coding
        self.mb2_leaky_relu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mb2_conv2d = nn.Conv2d(50, 100, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb2_batch_norm2d_1 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Third middle block coding
        self.mb3_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mb3_conv2d = nn.Conv2d(100, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb3_batch_norm2d_1 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Fourth middle block coding
        self.mb4_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mb4_conv2d = nn.Conv2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb4_batch_norm2d_1 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # # Fifth middle block coding
        # self.mb5_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.mb5_conv2d = nn.Conv2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # self.mb5_batch_norm2d_1 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # # Sixth middle block coding
        # self.mb6_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.mb6_conv2d = nn.Conv2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # self.mb6_batch_norm2d_1 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Middle block
        self.mb_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.mb_conv2d = nn.Conv2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb_relu = nn.ReLU(inplace=True)
        self.mb_conv_transpose2d = nn.ConvTranspose2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb_batch_norm2d = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # # Sixth middle block decoding
        # self.mb6_relu = nn.ReLU(inplace=True)
        # self.mb6_conv_transpose2d = nn.ConvTranspose2d(400, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # self.mb6_batch_norm2d_2 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.mb6_dropout = nn.Dropout(p=0.5, inplace=False)

        # # Fifth middle block decoding
        # self.mb5_relu = nn.ReLU(inplace=True)
        # self.mb5_conv_transpose2d = nn.ConvTranspose2d(400, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        # self.mb5_batch_norm2d_2 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.mb5_dropout = nn.Dropout(p=0.5, inplace=False)

        # Fourth middle block decoding
        self.mb4_relu = nn.ReLU(inplace=True)
        self.mb4_conv_transpose2d = nn.ConvTranspose2d(400, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb4_batch_norm2d_2 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mb4_dropout = nn.Dropout(p=0.5, inplace=False)

        # Third middle block decoding
        self.mb3_relu = nn.ReLU(inplace=True)
        self.mb3_conv_transpose2d = nn.ConvTranspose2d(400, 100, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb3_batch_norm2d_2 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Second middle block decoding
        self.mb2_relu = nn.ReLU(inplace=True)
        self.mb2_conv_transpose2d = nn.ConvTranspose2d(200, 50, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb2_batch_norm2d_2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # First middle block decoding
        self.mb1_relu = nn.ReLU(inplace=True)
        self.mb1_conv_transpose2d = nn.ConvTranspose2d(100, 25, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.mb1_batch_norm2d_2 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # First block decoding
        self.fb_relu = nn.ReLU(inplace=True)
        self.fb_conv_transpose2d = nn.ConvTranspose2d(50, 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.fb_batch_norm2d = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        def init_weights(layer):
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Init weights
        self.apply(init_weights)
    
    def forward(self, x):
        # First block coding
        x = self.fb_conv2d(x)
        fb_connection = x

        # First middle block coding
        x = self.mb1_leaky_relu(x)
        x = self.mb1_conv2d(x)
        x = self.mb1_batch_norm2d_1(x)
        mb1_connection = x

        # Second middle block coding
        x = self.mb2_leaky_relu(x)
        x = self.mb2_conv2d(x)
        x = self.mb2_batch_norm2d_1(x)
        mb2_connection = x

        # Third middle block coding
        x = self.mb3_leaky_relu(x)
        x = self.mb3_conv2d(x)
        x = self.mb3_batch_norm2d_1(x)
        mb3_connection = x

        # Fourth middle block coding
        x = self.mb4_leaky_relu(x)
        x = self.mb4_conv2d(x)
        x = self.mb4_batch_norm2d_1(x)
        mb4_connection = x

        # # Fifth middle block coding
        # x = self.mb5_leaky_relu(x)
        # x = self.mb5_conv2d(x)
        # x = self.mb5_batch_norm2d_1(x)
        # mb5_connection = x

        # # Sixth middle block coding
        # x = self.mb6_leaky_relu(x)
        # x = self.mb6_conv2d(x)
        # x = self.mb6_batch_norm2d_1(x)
        # mb6_connection = x

        # Middle block
        x = self.mb_leaky_relu(x)
        x = self.mb_conv2d(x)
        x = self.mb_relu(x)
        x = self.mb_conv_transpose2d(x)
        x = self.mb_batch_norm2d(x)

        # # Sixth middle block decoding
        # x = self.mb6_relu(torch.cat([mb6_connection, x], 1))
        # x = self.mb6_conv_transpose2d(x)
        # x = self.mb6_batch_norm2d_2(x)
        # x = self.mb6_dropout(x)

        # # Fifth middle block decoding
        # x = self.mb5_relu(torch.cat([mb5_connection, x], 1))
        # x = self.mb5_conv_transpose2d(x)
        # x = self.mb5_batch_norm2d_2(x)
        # x = self.mb5_dropout(x)

        # Fourth middle block decoding
        x = self.mb4_relu(torch.cat([mb4_connection, x], 1))
        x = self.mb4_conv_transpose2d(x)
        x = self.mb4_batch_norm2d_2(x)
        x = self.mb4_dropout(x)

        # Third middle block decoding
        x = self.mb3_relu(torch.cat([mb3_connection, x], 1))
        x = self.mb3_conv_transpose2d(x)
        x = self.mb3_batch_norm2d_2(x)

        # Second middle block decoding
        x = self.mb2_relu(torch.cat([mb2_connection, x], 1))
        x = self.mb2_conv_transpose2d(x)
        x = self.mb2_batch_norm2d_2(x)

        # First middle block decoding
        x = self.mb1_relu(torch.cat([mb1_connection, x], 1))
        x = self.mb1_conv_transpose2d(x)
        x = self.mb1_batch_norm2d_2(x)

        # First block decoding
        x = self.fb_relu(torch.cat([fb_connection, x], 1))
        x = self.fb_conv_transpose2d(x)
        x = self.fb_batch_norm2d(x)

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
