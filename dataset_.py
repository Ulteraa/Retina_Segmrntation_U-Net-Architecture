import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
class Retina(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        # self.mask_label = mask_label
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        #mask_path = os.path.join(self.mask_dir, self.images[index].replace(".tif", "_mask.gif"))

        mask_path = os.path.join(self.mask_dir, self.images[index].replace("_training.tif", "_manual1.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        #mask_label = np.array(Image.open(mask_label_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        #mask_label[mask_label == 255.0] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
if __name__=='__main__':
    data = Retina(image_dir='retina/DRIVE/training/images',
                  mask_dir='retina/DRIVE/training/1st_manual', transform=None)
    data_load = DataLoader(data, batch_size=1, shuffle=True)
    data_ = next(iter(data))
    plt.imshow(data_[0][:, :, 0])
    plt.show()
    print(data_[0].shape)
