# USAGE
# python train_resnet.py [1]
# path to folder containing training data

import os
import sys
import torch

from Utilities.local_dataset import make_dataloader
from Utilities.logger import Logger
from Utilities.model import Network
from Utilities.pretrain import get_pretrained

MODEL_FOLDER = "model"
MODEL_PATH = os.path.join(MODEL_FOLDER, "resnet-model.pt")


# maybe add some prints here
def train_model(model, train_data, val_data, epochs=10, display=200):
    data = next(iter(val_data))
    for epoch in range(epochs):
        loss_info = Logger.create_loss_data()
        loss_counter = 0
        for data in train_data:
            model.set_up_input(data)
            model.optimize()
            Logger.update_loss(model, loss_info, count=data['L'].size(0))
            loss_counter = loss_counter + 1
            if loss_counter % display == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Iteration {loss_counter}/{len(train_data)}")
                Logger.log_results(loss_info)
                Logger.visualize(model, data, save=False) 



args = sys.argv

if len(args) <= 1:
    print("No path provided - aborting!")
    exit(0)

path = args[1]

if not os.path.exists(path) and not os.path.isdir(path):
    print("Invalid path")
    exit(0)

print("Creating data loaders...")
train = make_dataloader(os.path.join(path, "bnw"))
validate = make_dataloader(os.path.join(path, "colour"))

print("Seraching for device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Got device: " + str(device))

print("Getting pretrained resnet...")
resnet = get_pretrained(train, device)

print("Building model...")
model = Network(device, resnet=resnet)

print("Training model...")
train_model(model, train, validate)

print("Saving model...")
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

torch.save(model.state_dict(), MODEL_PATH)

print("Finished")
