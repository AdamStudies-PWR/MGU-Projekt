import numpy as np
import os
import shutil
import sys
import torch

from torch import nn
from PIL import Image

from models import Unet
from skimage.color import lab2rgb
from torchvision import transforms

MODEL_SAVE_PATH = 'model/model.pt'
RESULT_PATH = 'result/'


def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


args = sys.argv

if len(args) <= 1:
    print("No path to image provided")
    exit(0)

img_path = args[1]
if not os.path.exists(img_path) and not os.path.isfile(img_path):
    print("Invalid path - not an image")
    exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

img = Image.open(img_path)
img = img.resize((256, 256))
tensor = transforms.ToTensor()(img)[:1] * 2. - 1.

model.eval()
with torch.no_grad():
    preds = model(tensor.unsqueeze(0).to(device))

colorized = lab_to_rgb(tensor.unsqueeze(0), preds.cpu())[0]

if os.path.exists(RESULT_PATH):
    shutil.rmtree(RESULT_PATH)
os.mkdir(RESULT_PATH)

filename = os.path.basename(img_path)
filename = os.path.join(RESULT_PATH, filename)

colorized = Image.fromarray(np.uint8(colorized * 255))
colorized.save(filename + "-colorized.png")
