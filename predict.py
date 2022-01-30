from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.nn import functional as F
from pathlib import Path
from utils import DiceLoss


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)
args = parser.parse_args()

tfms = transforms.Compose([
    transforms.ToTensor(),
])

if __name__ == '__main__':
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes)
    net.load_from(config)
    net.eval()

    MODE = 'EVAL'
    n_classes = 4

    dice_loss = DiceLoss(n_classes)
    conf_thresh = 0.9

    for img_path in glob.glob(os.path.join(args.root_path, 'images', '*')):
        img_np = np.load(img_path)
        img_np = np.pad(img_np, (8,), 'constant')
        if np.amax(img_np) - np.amin(img_np) != 0:
            img_np = (img_np - np.amin(img_np)) / (np.amax(img_np) - np.amin(img_np))
        img_pil = Image.fromarray(img_np)
        img = tfms(img_pil)
        out = net(img.unsqueeze(0)).squeeze(0)
        out = out[:, 8: -8, 8: -8]
        out = F.softmax(out, dim=0)

        if MODE == 'EVAL':
            img_name = Path(img_path).name
            lbl_name = img_name.replace("_flair", "").replace("_t1ce", "").replace("_t1", "").replace("_t2", "")
            lbl_path = os.path.join(args.root_path, 'labels', lbl_name)
            lbl = np.load(lbl_path)
            lbl = lbl.clip(0, 3).astype(np.int64)
            lbl = torch.from_numpy(lbl)
            # lbl = F.one_hot(lbl)
            # lbl = lbl.permute(2, 0, 1)

            dl = dice_loss(out.unsqueeze(0), lbl.unsqueeze(0))

            print(dl)
        print()
