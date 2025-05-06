import torch
import torch.nn.functional as F
import random
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from dataloaders.dataset import DataSet, RandomGenerator
from scipy import ndimage
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from monai import transforms
import torchvision.transforms.functional as TF
from utils.bezier import bezier_curve
import numpy as np

def norm(slices):
    out = 2 * slices - 1
    return out

def denorm(slices):
    out = (slices + 1) / 2
    return out

def nonlinear_transformation(slices):
    slices = slices.cpu().numpy()
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]

    xvals, yvals = bezier_curve(points, nTimes=100000)
    # xvals = np.sort(xvals_2)
    # yvals = np.sort(yvals_2)
    # if random.random() < 0.5:
    #     # Half change to get flip
    #     xvals = np.sort(xvals)
    # else:
    xvals, yvals = np.sort(xvals), np.sort(yvals)

    """
    slices, nonlinear_slices_2, nonlinear_slices_4 are source-similar images
    nonlinear_slices_1, nonlinear_slices_3, nonlinear_slices_5 are source-dissimilar images
    """
    slices = np.interp(slices, xvals, yvals)

    # slices = denorm(slices)
    slices = torch.from_numpy(slices.astype(np.float32)).cuda()
    return slices


class strong_weak_augmentation():
    def __init__(self, crop_size=(96,96)):
        super(strong_weak_augmentation, self).__init__()

        self.scale_x = 0.9 + 0.3 * random.random()
        self.scale_y = 0.9 + 0.3 * random.random()
        # self.angle = random.uniform(-90, 90)

        self.crop_w = crop_size[0]
        self.crop_h = crop_size[1]

    def init_scale_rotate(self):
        self.scale_x = 0.9 + 0.2 * random.random()
        self.scale_y = 0.9 + 0.2 * random.random()
        # self.angle = random.uniform(-20, 20)
        # print(f'init: sacle ({self.scale_x},{self.scale_y}), angle ({self.angle})')

    def aug(self, image):
        # image = TF.rotate(image, self.angle)  # 不改变大小 B,C,H,W
        # image = F.interpolate(image, scale_factor=(self.scale_x, self.scale_y), mode='bilinear',
        #                       align_corners=False)
        # image = ndimage.rotate(image, self.angle, order=0, reshape=False)
        # image = TF.rotate(image, self.angle)
        image = F.interpolate(image, scale_factor=(self.scale_x, self.scale_y), mode='bilinear',
                                                    align_corners=False)
        w, h = image.shape[2], image.shape[3]
        if w < self.crop_w:
            # padding width
            left_padding = (self.crop_w - w) // 2
            right_padding = self.crop_w - w - left_padding
        else:
            left_padding = 0
            right_padding = 0
        if h < self.crop_h:
            top_padding = (self.crop_h - h) // 2
            bottom_padding = self.crop_h - h - top_padding
        else:
            top_padding = 0
            bottom_padding = 0
        image = F.pad(image, (top_padding, bottom_padding, left_padding, right_padding))

        w, h = image.shape[2], image.shape[3]
        start_x = (w - self.crop_w) // 2
        start_y = (h - self.crop_h) // 2
        image = image[:, :, start_x:start_x + self.crop_w, start_y:start_y + self.crop_h]
        return image

    def strong_augment(self, image, label=None, add_noise=True):

        image = self.aug(image)
        image = nonlinear_transformation(image)
        if add_noise and random.random()>0.5:
            noise = torch.randn_like(image) * 0.1
            image = torch.clamp(image + noise, 0, 1)

        if label !=None:
            label = label.unsqueeze(1)
            label = self.aug(label)
            label = label.squeeze(1)
            return image, label
        else:
            return image

    def weak_augment(self, image, label=None):

        image = self.aug(image)

        if label != None:
            label = label.unsqueeze(1)
            label = self.aug(label)
            label = label.squeeze(1)
            return image, label
        else:
            return image


if __name__ == '__main__':
    db = DataSet(split_dir='../data/abdomen', data_dir='D://data/Multimodal/abdomen_l', modality='T1in', split='train',
                          transform=RandomGenerator((128,128)))

    train_loader = DataLoader(db, batch_size=4, shuffle=True,
                              num_workers=1, pin_memory=True)
    num = 450
    sample = db.__getitem__(num)
    image = sample['image']
    label = sample['label']
    image = image.expand(4, -1, -1, -1)
    label = label.repeat(4, 1, 1)

    sw_aug = strong_weak_augmentation((128,128))
    sw_aug.init_scale_rotate()

    image_aug, label_aug = sw_aug.strong_augment(image, label)
    image_re = sw_aug.weak_augment(image)

    fig, axes = plt.subplots(2, 3)
    axes[0, 0].imshow(image_aug[0].squeeze().cpu(), cmap='gray')
    axes[0, 0].set_title('Strong Aug')
    axes[0, 1].imshow(image_re[0].squeeze().cpu(), cmap='gray')
    axes[0, 1].set_title('Weak Aug')
    axes[0, 2].imshow(label_aug[0].squeeze().cpu(), cmap='gray')
    axes[0, 2].set_title('Label Aug')
    axes[1, 0].imshow(image[0].squeeze().cpu(), cmap='gray')
    axes[1, 0].set_title('Origin Image')
    axes[1, 1].imshow(image[0].squeeze().cpu(), cmap='gray')
    axes[1, 1].set_title('Origin Image')
    axes[1, 2].imshow(label[0].squeeze().cpu(), cmap='gray')
    axes[1, 2].set_title('Origin Label')
    plt.tight_layout()
    plt.show()

    # plt.imsave('./origin.png', image[0].squeeze().cpu(), cmap='gray')
    # plt.imsave('./weak_aug.png', image_re[0].squeeze().cpu(), cmap='gray')
    # plt.imsave('./strong_aug.png', image_aug[0].squeeze().cpu(), cmap='gray')