# USAGE
# python train_resnet.py [1] [2] [3]
# [1] - path to folder containing training data (black nad white)
# [2] - path to folder containing validation data (colour)
# [3] - (optional, default=2) 1 - use pretrained resnet | 2 use custom unet

import os
import torch
import sys

from Utils.dataloaders import make_dataloaders
from Utils.models import MainModel
from Utils.utils import update_losses, log_results, visualize, create_loss_meters
from Utils.pretrain import get_pretrained


MODEL_PATH = "./model"
MODEL_NAME = "/model.pt"
MODEL_NAME_RESNET = "/model-resnet18.pt"


def train_model(model, train_dl, val_dl, epochs, display_every=200):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in train_dl:
            model.setup_input(data) 
            model.optimize()
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

mode = "2"
if len(args) == 4:
    mode = args[3]

print("Building model...")
model = None
result_path = ""
if mode == "2":
    result_path = MODEL_NAME
    model = MainModel()
else:
    result_path = MODEL_NAME_RESNET
    pretrained = get_pretrained(train_dl)
    model = MainModel(net_G=pretrained)

print("Training model...")
if mode == "2":
    train_model(model, train_dl, val_dl, 100)
else:
    train_model(model, train_dl, val_dl, 20)

print("Saving data...")
if os.path.exists(MODEL_PATH + result_path):
    os.remove(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

torch.save(model.state_dict(), MODEL_PATH + result_path)

print("Finished")
