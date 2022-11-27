# USAGE
# python colorize.py [1] [2] [3]
# [1] - path to image
# [2] - path to saved model weights
# [3] - mode | 1 - Unet from Submodules, 2 - Pretrained Resnet18 | optional | default = 1

import os
import shutil
import sys
import torch

import numpy as np

from PIL import Image
from skimage.color import lab2rgb
from torchvision import transforms

from Utilities.model import Network
from Utilities.pretrain import get_resnet


from matplotlib import pyplot as plt


RESULT_PATH = "result"


def get_images(batch, cpu):
    batch = (batch + 1.) * 50.
    cpu = cpu * 110.
    batch = torch.cat([batch, cpu], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    images = []
    for img in batch:
        img = lab2rgb(img)
        images.append(img)
    return np.stack(images, axis=0)


args = sys.argv

if len(args) <= 2:
    print("No path provided - aborting!")
    exit(0)

path = args[1]
model_path = args[2]

if not os.path.exists(path) and not os.path.isfile(path):
    print("Invalid path")
    exit(0)

if not os.path.exists(model_path) and not os.path.isfile(model_path):
    print("Invalid model file - run train.py first")
    exit(0)

version = 1
if len(args) >= 4:
    version = args[3]

print("Seraching for device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Got device: " + str(device))

print("Loading model...")
resnet = None
if version == "2":
    resnet = get_resnet(device)

model = Network(device, resnet=resnet)
model.load_state_dict(torch.load(model_path, map_location=device))

img = Image.open(path)
tensor = transforms.ToTensor()(img)[:1] * 2. - 1.
model.eval()

print("Colorizning image...")
with torch.no_grad():
    predicates = model.unet(tensor.unsqueeze(0).to(device))
colorized = get_images(tensor.unsqueeze(0), predicates.cpu())[0]

print("Saving result...")

if os.path.exists(RESULT_PATH):
    shutil.rmtree(RESULT_PATH)
os.mkdir(RESULT_PATH)

filename = os.path.basename(path)
filename = os.path.join(RESULT_PATH, filename)

plt.imshow(colorized)

colorized = Image.fromarray(np.uint8(colorized * 255))
colorized.save(filename + "-colorized.png")

print("Finished")
