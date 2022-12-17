import numpy as np
import os

from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


SIZE = 64


class LocalDataset(Dataset):
    def __init__(self, folder, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  transforms.InterpolationMode.BICUBIC)

        self.dataset = [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isfile(
            os.path.join(folder, file))]
    
    def __getitem__(self, idx):
        img = Image.open(self.dataset[idx]).convert("RGB")
        img = np.array(img)
        lab = rgb2lab(img).astype("float32") # Conversion to Lab colour space
        tensor = transforms.ToTensor()(lab)
        L = tensor[[0], ...] / 50. - 1.
        AB = tensor[[1, 2], ...] / 110.

        return {'L': L, 'AB': AB}
    
    def __len__(self):
        return len(self.dataset)


def make_DataLoader(batch_size=16, n_workers=4, pin_memory=True, **kwargs):
    dataset = LocalDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader
