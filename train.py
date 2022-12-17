import os
import torch


from Network.model import Model
from Utils.dataloaders import make_DataLoader


TRAIN_PATH = "./dataset/train/black_and_white"
VAL_PATH = "./dataset/train/colour"


if not os.path.exists(TRAIN_PATH) or not os.path.exists(VAL_PATH):
    print("Invalid training data!")
    exit(0)

print("Setting up dataloaders...")
train_data = make_DataLoader(folder=TRAIN_PATH, split="train")
val_data = make_DataLoader(folder=VAL_PATH, split="val")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Init device: " + str(device))

print("Setting up model...")
model = Model(device)
