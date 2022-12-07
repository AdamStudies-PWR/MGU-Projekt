# USAGE
# python train_resnet.py [1] [2]
# [1] - path to black and white image
# [2] - (mode), 1 - use with resnet18 | 2 use custom unet

import numpy as np
import os
import PIL
import shutil
import sys
import torch

from torchvision import transforms

from Utils.models import MainModel
from Utils.utils import lab_to_rgb
from Utils.pretrain import get_model


RESULT_PATH = "result"
MODEL_NAME = "model/model.pt"
MODEL_NAME_RESNET = "model/model-resnet18.pt"


args = sys.argv
if len(args) <= 2:
    print("Not enough arguments")
    exit(0)

image_path = args[1]
if not os.path.exists(image_path) and not os.path.isfile(image_path):
    print("Invalid path - not an image")
    exit(0)

mode = args[2]
model = None
use_path = ""

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if mode == "2":
    print("Mode: Custom Unet")
    model = MainModel()
    use_path = MODEL_NAME
else:
    print("Mode: Resnet")
    resnet = get_model(device)
    model = MainModel(net_G=resnet)
    use_path = MODEL_NAME_RESNET

model.load_state_dict(
    torch.load(
        use_path,
        map_location=device
    )
)

print("Colorizing image...")
img = PIL.Image.open(image_path)
img = img.resize((256, 256))
# to make it between -1 and 1
img = transforms.ToTensor()(img)[:1] * 2. - 1.
model.eval()
with torch.no_grad():
    preds = model.net_G(img.unsqueeze(0).to(device))
colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]

print("Saving result...")

if os.path.exists(RESULT_PATH):
    shutil.rmtree(RESULT_PATH)
os.mkdir(RESULT_PATH)

filename = os.path.basename(image_path)
filename = os.path.join(RESULT_PATH, filename)

colorized = PIL.Image.fromarray(np.uint8(colorized * 255))
colorized.save(filename + "-colorized.png")

print("Finished")
