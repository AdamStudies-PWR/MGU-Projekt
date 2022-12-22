# USAGE
# python train_resnet.py [1] [2]
# [1] - path to folder containing training data (black nad white)
# [2] - path to folder containing validation data (colour)

import os
import torch
import sys

from torch import nn, optim

from Utils.dataloaders import make_dataloaders
from Utils.utils import update_losses, log_results, visualize, create_loss_meters
from Utils.loss import GANLoss


MODEL_PATH = "./model"
MODEL_NAME = "/model.pt"


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

      # Fifth middle block coding
      self.mb5_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
      self.mb5_conv2d = nn.Conv2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      self.mb5_batch_norm2d_1 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      # Sixth middle block coding
      self.mb6_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
      self.mb6_conv2d = nn.Conv2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      self.mb6_batch_norm2d_1 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      # Middle block
      self.mb_leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
      self.mb_conv2d = nn.Conv2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      self.mb_relu = nn.ReLU(inplace=True)
      self.mb_conv_transpose2d = nn.ConvTranspose2d(200, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      self.mb_batch_norm2d = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

      # Sixth middle block decoding
      self.mb6_relu = nn.ReLU(inplace=True)
      self.mb6_conv_transpose2d = nn.ConvTranspose2d(400, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      self.mb6_batch_norm2d_2 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      self.mb6_dropout = nn.Dropout(p=0.5, inplace=False)

      # Fifth middle block decoding
      self.mb5_relu = nn.ReLU(inplace=True)
      self.mb5_conv_transpose2d = nn.ConvTranspose2d(400, 200, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      self.mb5_batch_norm2d_2 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      self.mb5_dropout = nn.Dropout(p=0.5, inplace=False)

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

      # Fifth middle block coding
      x = self.mb5_leaky_relu(x)
      x = self.mb5_conv2d(x)
      x = self.mb5_batch_norm2d_1(x)
      mb5_connection = x

      # Sixth middle block coding
      x = self.mb6_leaky_relu(x)
      x = self.mb6_conv2d(x)
      x = self.mb6_batch_norm2d_1(x)
      mb6_connection = x

      # Middle block
      x = self.mb_leaky_relu(x)
      x = self.mb_conv2d(x)
      x = self.mb_relu(x)
      x = self.mb_conv_transpose2d(x)
      x = self.mb_batch_norm2d(x)

      # Sixth middle block decoding
      x = self.mb6_relu(torch.cat([mb6_connection, x], 1))
      x = self.mb6_conv_transpose2d(x)
      x = self.mb6_batch_norm2d_2(x)
      x = self.mb6_dropout(x)

      # Fifth middle block decoding
      x = self.mb5_relu(torch.cat([mb5_connection, x], 1))
      x = self.mb5_conv_transpose2d(x)
      x = self.mb5_batch_norm2d_2(x)
      x = self.mb5_dropout(x)

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

    def forward(self, x):
        return self.model(x)


def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print("Model " +  net.__class__.__name__ +  f" initialized with {init} initialization")
    return net


class MainModel(nn.Module):
    def __init__(self, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        self.net_G = init_weights(Unet().to(device))
        self.net_D = init_weights(Discriminator().to(device))
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def optimize(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

        # Train discriminator
        self.fake_color = self.net_G(self.L)
        self.net_D.train()
        for p in net_D.parameters():
            p.requires_grad = True
        self.opt_D.zero_grad()
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        self.opt_D.step()

        # Train Generator
        self.net_G.train()
        for p in net_D.parameters():
            p.requires_grad = False
        self.opt_G.zero_grad()
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        self.opt_G.step()


def train_model(model, train_dl, val_dl, epochs, display_every=2):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in train_dl:
            model.optimize(data)
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs


args = sys.argv
if len(args) <= 2:
    print("Not enough arguments. Please provide train and validate datasets path")
    exit(0)

train_paths = args[1]
val_paths = args[2]

if not os.path.exists(train_paths) or not os.path.isdir(train_paths):
    print("Invalid train path")
    exit(0)

if not os.path.exists(val_paths) or not os.path.isdir(val_paths):
    print("Invalid validation path")
    exit(0)

print("Got train path: " + train_paths)
print("Got validate path: " + val_paths)

print("Preparing data loaders...")
train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))

print("Building model...")
result_path = MODEL_NAME
model = MainModel()

print("Training model...")
train_model(model, train_dl, val_dl, 100)


print("Saving data...")
if os.path.exists(MODEL_PATH + result_path):
    os.remove(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

torch.save(model.state_dict(), MODEL_PATH + result_path)

print("Finished")
