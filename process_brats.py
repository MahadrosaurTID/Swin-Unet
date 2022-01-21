import numpy as np
import nibabel as nb
import glob
import os
from pathlib import Path


if __name__ == '__main__':
    base_dir = 'brats_dataset'
    out_dir = 'brats_dataset_processed'
    ext = '.nii.gz'

    try:
        os.mkdir(os.path.join(out_dir, 'images'))
    except FileExistsError:
        pass
    try:
        os.mkdir(os.path.join(out_dir, 'labels'))
    except FileExistsError:
        pass

    img_dir_paths = glob.glob(os.path.join(base_dir, 'images', '*'))

    for img_dir in img_dir_paths:
        filenames = os.listdir(img_dir)
        for filename in filenames:
            dat = nb.load(os.path.join(img_dir, filename)).get_fdata()
            for i in range(dat.shape[2]):
                dat_slice = dat[:, :, i]
                out_path = os.path.join(out_dir, 'images', filename.replace(ext, "_")+str(i)+".npy")
                np.save(out_path, dat_slice)

    label_dir_paths = glob.glob(os.path.join(base_dir, 'labels', '*'))
    for label_dir in label_dir_paths:
        filenames = os.listdir(label_dir)
        for filename in filenames:
            dat = nb.load(os.path.join(label_dir, filename)).get_fdata()
            for i in range(dat.shape[2]):
                dat_slice = dat[:, :, i]
                out_path = os.path.join(out_dir, 'labels', filename.replace(ext, "_")+str(i)+".npy")
                np.save(out_path, dat_slice)
