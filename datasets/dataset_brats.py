import torch
from torch.utils.data import Dataset
import glob
import os
import nibabel as nb


class brats_dataset_2d(Dataset):

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.all_vols = glob.glob(os.path.join(base_dir, '*', '*.nii')) + glob.glob(os.path.join(base_dir, '*', '*.nii.gz'))
        self.label_list = [lbl for lbl in self.all_vols if "_seg.nii" in lbl]
        self.img_list = [img for img in self.all_vols if img not in self.label_list]

    def __getitem__(self, index):
        img = nb.load(self.img_list[index])
        img = img.get_fdata()
        img = img[:, :, 0:3]
        img = torch.tensor(img, dtype=torch.float)
        img = img.resize_((224, 224, 3))
        img = img.permute(2, 0, 1)

        label_name = '_'.join(self.img_list[index].split('_')[:-1]) + "_seg.nii.gz"
        label = nb.load(label_name)
        label = label.get_fdata()
        label = label[:, :, 0]
        label = torch.tensor(label, dtype=torch.int)
        label = label.resize_((224, 224))
        # label = label.permute(2, 0, 1)

        # print("img", img)
        # print("label", label)
        return {'image': img, 'label': label}

    def __len__(self):
        return len(self.label_list)
