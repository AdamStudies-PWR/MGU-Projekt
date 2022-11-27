import os
import torch

from alive_progress import alive_bar
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torch import nn, optim
from torchvision import models


MODEL_FOLDER = "model"
MODEL_PATH = os.path.join(MODEL_FOLDER, "resnet-pretrained.pt")


def build_res_unet(device, n_input=1, n_output=2, size=100):
    body = create_body(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), pretrained=True, n_in=n_input, cut=-2)
    resnet = DynamicUnet(body, n_output, (size, size)).to(device)
    return resnet   


def pretrain_resnet(model, train_data, device, epochs=20):
    opt = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    for i in range(epochs):
        print("[" + str(i + 1) + "/" + str(epochs) + "]")
        with alive_bar(len(train_data)) as bar:
            for data in train_data:
                L, ab = data['L'].to(device), data['ab'].to(device)
                predicates = model(L)
                loss = criterion(predicates, ab)
                opt.zero_grad()
                loss.backward()
                opt.step()
                bar()


def gen_or_load(train_data, device):
    resnet = build_res_unet(device)
    if os.path.exists(MODEL_PATH):
        print("Pretrained state found - loading...")
        resnet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Pretraining resnet...")
        pretrain_resnet(resnet, train_data, device)
        print("Saving state...")
        torch.save(resnet.state_dict(), MODEL_PATH)

    return resnet


def get_pretrained(train_data, device):
    return gen_or_load(train_data, device)


def get_resnet(device):
    resnet = build_res_unet(device)

    return resnet
