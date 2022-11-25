import os
import shutil
import sys
import torch

from alive_progress import alive_bar
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torchvision.models.resnet import resnet18

from Utilities.local_dataset import make_dataloader
from Utilities.model import Network


MODEL_FOLDER = "model"
MODEL_PATH = os.path.join(MODEL_FOLDER, "resnet-model.pt")


# maybe add some prints here
def train_model(model, train_data, val_data, epochs=100):
    data = next(iter(val_data))
    for i in range(epochs):
        print("[" + str(i) + "/" + str(epochs) + "]")
        with alive_bar(len(train_data)) as bar:
            for data in train_data:
                model.set_up_input(data)
                model.optimize()
                bar()


def build_res_unet(n_input=1, n_output=2, size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    resnet = DynamicUnet(body, n_output, (size, size)).to(device)
    return resnet   


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

print("Building model...")
resnet = build_res_unet()
model = Network(resnet=resnet)

print("Training model...")
train_model(model, train, validate)

print("Saving model...")
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

torch.save(model.state_dict(), MODEL_PATH)

print("Finished")
