import os
import torch
import sys
import numpy as np

from torch import nn, optim
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models import Unet, Discriminator

# Constants that need to be set
MODEL_SAVE_PATH = 'model/'
DATASET_PATH = 'dataset/train/' # Dir with 256x256 images
EPOCHS = 100

# Dataset that load images in lab format
class ColorizationDataset(Dataset):
    def __init__(self, dataset_path):
        self.paths = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path)]
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)

# Prepare dataset loader
dataset = ColorizationDataset(DATASET_PATH)
data_loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

# Create models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Unet().to(device)
discriminator = Discriminator().to(device)

# Create loss function
loss_function = nn.BCEWithLogitsLoss().to(device)
l1_loss = nn.L1Loss().to(device)

# Create optimizers
generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Train
for e in range(EPOCHS):
    i = 0                                 
    for data in data_loader:
        # Fetch image channels from data
        L = data['L'].to(device)
        ab = data['ab'].to(device)

        # Train discriminator
        fake_color = generator(L)
        discriminator.train()
        for param in discriminator.parameters():
            param.requires_grad = True
        discriminator_optimizer.zero_grad()
        fake_image = torch.cat([L, fake_color], dim=1)
        fake_preds = discriminator(fake_image.detach())
        discriminator_loss_fake = loss_function(fake_preds, torch.tensor(1.0).expand_as(fake_preds).to(device))
        real_image = torch.cat([L, ab], dim=1)
        real_preds = discriminator(real_image)
        discriminator_loss_real = loss_function(real_preds, torch.tensor(0.0).expand_as(real_preds).to(device))
        discriminator_loss = (discriminator_loss_fake + discriminator_loss_real) * 0.5
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train Generator
        generator.train()
        for param in discriminator.parameters():
            param.requires_grad = False
        generator_optimizer.zero_grad()
        fake_image = torch.cat([L, fake_color], dim=1)
        fake_preds = discriminator(fake_image)
        generator_loss = loss_function(fake_preds, torch.tensor(0.0).expand_as(fake_preds).to(device))
        generator_l1_loss = l1_loss(fake_color, ab) * 100
        generator_loss = generator_loss + generator_l1_loss
        generator_loss.backward()
        generator_optimizer.step()
        
        # Print train state
        i += 1
        print(f"\nEpoch {e+1}/{EPOCHS}")
        print(f"Iteration {i}/{len(data_loader)}")

# Save generator
torch.save(generator.state_dict(), os.path.join(MODEL_SAVE_PATH, 'model.pt'))
