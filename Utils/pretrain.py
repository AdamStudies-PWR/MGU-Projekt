import os
import torch

from alive_progress import alive_bar
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torch import nn, optim
from torchvision import models

from Utils.utils import AverageMeter


MODEL_PATH = "./model"


def build_res_unet(device, n_input=1, n_output=2, size=256):
    body = create_body(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1), pretrained=True, n_in=n_input,
        cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G


def pretrain_generator(device, net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        with alive_bar(len(train_dl)) as bar:
            for data in train_dl:
                L, ab = data['L'].to(device), data['ab'].to(device)
                preds = net_G(L)
                loss = criterion(preds, ab)
                opt.zero_grad()
                loss.backward()
                opt.step()
            
                loss_meter.update(loss.item(), L.size(0))
                bar()
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")


def get_model(device):
    net_G = build_res_unet(device, n_input=1, n_output=2, size=256)
    return net_G


def get_pretrained(train_dl):
    print("Get pretrained...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_G = get_model(device)

    if os.path.exists(MODEL_PATH + "/res18-unet.pt"):
        print("Loading pretrained state...")
        net_G.load_state_dict(torch.load(MODEL_PATH + "/res18-unet.pt", map_location=device))
    else:
        print("Pretraining resnet...")
        opt = optim.Adam(net_G.parameters(), lr=1e-4)
        criterion = nn.L1Loss()        
        pretrain_generator(device, net_G, train_dl, opt, criterion, 20)
        print("Saving pretraining result for future use...")
        if not os.path.exists(MODEL_PATH):
            os.mkdir(MODEL_PATH)
        torch.save(net_G.state_dict(), MODEL_PATH + "/res18-unet.pt")

    return net_G
