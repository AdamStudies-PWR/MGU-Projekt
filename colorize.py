import os
import sys
import torch

from PIL import Image
from skimage.color import lab2rgb
from torchvision import transforms

from Utilities.model import Network


MODEL_FOLDER = "model"
MODEL_PATH = os.path.join(MODEL_FOLDER + "model.pt")
RESULT_PATH = "result"


def get_images(batch, cpu):
    batch = (batch + 1.) * 50.
    cpu = cpu * 110.
    images = []
    for img in batch:
        img = lab2rgb(img)
        images.append(img)
    return images


args = sys.argv

if len(args) <= 1:
    print("No path provided - aborting!")
    exit(0)

path = args[1]

if not os.path.exists(path) and not os.path.isfile(path):
    print("Invalid path")
    exit(0)

if not os.path.exists(MODEL_PATH) and not os.path.isfile(MODEL_PATH):
    print("Model does not exist - run train.py first")
    exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
model = Network()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

img = Image.open(path)
tensor = transforms.ToTensor()(img)[:1] * 2. - 1.
model.eval()

print("Colorizning image...")
with torch.no_grad():
    predicates = model.unet(tensor.unsqueeze(0).to(device))
colorized = get_images(tensor.unsqueeze(0), predicates.cpu())

print("Saving result...")
filename = os.path.basename(path)
filename = os.path.join(RESULT_PATH, filename)
counter = 0
for img in colorized:
    img.save(os.path.join(filename, "-colorized" + counter + ".png"))
    counter = counter + 1

print("Finished")
