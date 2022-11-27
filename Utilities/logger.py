import matplotlib.pyplot as plt
import time
import torch

import numpy as np
from skimage.color import rgb2lab, lab2rgb


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


class Logger:
    def create_loss_data():
        gen_gan_loss = AverageMeter()
        train_gan_loss = AverageMeter()
        gan_loss = AverageMeter()
        gan_loss = AverageMeter()
        loss_loss = AverageMeter()
        loss_combined = AverageMeter()

        return {'gen_gan_loss': gen_gan_loss,
            'train_gan_loss': train_gan_loss,
            'gan_loss': gan_loss,
            'gan_loss': gan_loss,
            'loss_loss': loss_loss,
            'loss_combined': loss_combined}
    
    def update_loss(model, loss_meter_dict, count):
        for loss_name, loss_meter in loss_meter_dict.items():
            loss = getattr(model, loss_name)
            loss_meter.update(loss.item(), count=count)

    def log_results(loss_meter_dict):
        for loss_name, loss_meter in loss_meter_dict.items():
            print(f"{loss_name}: {loss_meter.avg:.5f}")
    
    def visualize(model, data, save=True):
        model.unet.eval()
        with torch.no_grad():
            model.set_up_input(data)
            model.forward()
        model.unet.train()
        gen_colours = model.gen_colours.detach()
        real_color = model.ab
        L = model.L
        fake_imgs = lab_to_rgb(L, gen_colours)
        real_imgs = lab_to_rgb(L, real_color)
        fig = plt.figure(figsize=(15, 8))
        for i in range(5):
            ax = plt.subplot(3, 5, i + 1)
            ax.imshow(L[i][0].cpu(), cmap='gray')
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 5)
            ax.imshow(fake_imgs[i])
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 10)
            ax.imshow(real_imgs[i])
            ax.axis("off")
        # plt.show()
        if save:
            fig.savefig(f"colorization_{time.time()}.png")
