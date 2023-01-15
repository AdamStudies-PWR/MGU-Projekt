import timm
import torch
model = timm.create_model('inception_resnet_v2', pretrained=True)
model.eval()
import csv
import os
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.preprocessing import MinMaxScaler

import torch
use_gpu = torch.cuda.is_available()
if use_gpu:
    model.cuda()
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
PATH="../datasets/cifar10/train/"
PATH2="../datasets/cifar10/val/"
all_files = [os.path.join(PATH, file) for file in os.listdir(PATH) if os.path.isfile(
            os.path.join(PATH, file))]
all_files2 = [os.path.join(PATH2, file) for file in os.listdir(PATH2) if os.path.isfile(
            os.path.join(PATH2, file))]
all_files = all_files + all_files2

do_softmax = False

def save_to_csv(data):
    with open("embeddings-cifar10-normalization.csv", "w", encoding="UTF8", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def normalize(data):
    filenames = [i[0] for i in data]
    features = [i[1:] for i in data]
    scaler = MinMaxScaler()
    data2 = scaler.fit_transform(features).tolist()
    for index, i in enumerate(data2):
        i.insert(0, filenames[index])
    return data2

data_to_save = []
for index, i in enumerate(all_files):
    img = Image.open(i).convert('RGB')
    tensor = transform(img).unsqueeze(0) # transform and add batch dimension
    with torch.no_grad():
        out = model(tensor.cuda() if use_gpu else tensor )
    if do_softmax:
        probabilities = torch.nn.functional.softmax(out[0], dim=0).cpu().numpy().tolist()
    else:
        probabilities = out.cpu().numpy().tolist()
        probabilities = probabilities[0]

    probabilities.insert(0, os.path.basename(i))
    data_to_save.append(probabilities)

if not do_softmax:
    data_to_save = normalize(data_to_save)

save_to_csv(data_to_save)

 