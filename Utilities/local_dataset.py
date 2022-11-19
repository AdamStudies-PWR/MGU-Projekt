import os

import numpy as np

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


class LocalDataset():
    def __init__(self, path):
        self.paths = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(
            os.path.join(path, file))]

    def load_img(self, img_path):
        img = Image.open(img_path)
        img = np.array(img, dtype=float)
        img = transforms.ToTensor()(img)
        return img

    def __getitem__(self, idx):
        img = self.load_img(self.paths[idx])
        L = img[[0], ...] / 50. -1.
        ab = img[[1, 2], ...] / 110.
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)


def make_dataloader(path):
    dataset = LocalDataset(path)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

    return dataloader
