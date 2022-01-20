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

    img_dir_paths = glob.glob(os.path.join(base_dir, '*'))
    for img_dir in img_dir_paths:
        filenames = os.listdir(img_dir)
        for filename in filenames:
            dat = nb.load(os.path.join(img_dir, filename)).get_fdata()
            for i in range(dat.shape[2]):
                dat_slice = dat[:, :, i]
                if "_seg"+ext in filename:
                    out_path = os.path.join(out_dir, 'labels', filename.replace("_seg"+ext, "_")+str(i)+".npy")
                else:
                    out_path = os.path.join(out_dir, 'images', filename.replace(ext, "_")+str(i)+".npy")
                np.save(out_path, dat_slice)
