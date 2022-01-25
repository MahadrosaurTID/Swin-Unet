import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nb
import glob
import os
from pathlib import Path, PurePath
import numpy as np
from torchvision.transforms import transforms
from torchvision.transforms import ToTensor
from PIL import Image

tfms = transforms.Compose([
    # transforms.ColorJitter(brightness=(0.8, 1.25), contrast=(0.8, 1.4)),
    transforms.RandomAffine(degrees=(-10, 10),
                            shear=(-10, 10, -10, 10)),
    transforms.ToTensor(),
])


class brats_dataset(Dataset):

    def __init__(self, base_dir, tfms=tfms):
        self.base_dir = base_dir
        self.tfms = tfms
        self.img_paths = glob.glob(os.path.join(base_dir, 'images', '*.npy'))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = np.load(img_path)
        img = np.pad(img, (8,), 'constant')
        m = np.amin(img)
        mm = np.amax(img)
        if mm - m != 0:
            img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

        img_pil = Image.fromarray(img)

        lbl_path = img_path.replace("/images/", "/labels/").replace("_flair", "").replace("_t1ce", "").replace("_t1", "").replace("_t2", "")
        lbl = np.load(lbl_path)
        lbl = np.pad(lbl, (8,), 'constant')
        lbl = lbl.clip(0, 3).astype(np.uint8)

        # img_t = torch.tensor(img).unsqueeze(0)

        lbl_t = torch.tensor(lbl)
        return self.tfms(img_pil), self.tfms(img_pil), lbl_t

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    ds = brats_dataset('../brats_dataset_processed')
    dl = DataLoader(ds)
    for img, lbl in dl:
        print(torch.max(img))
