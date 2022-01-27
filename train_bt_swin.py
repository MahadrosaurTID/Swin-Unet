from datasets.dataset_brats import brats_dataset
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt
from networks.vision_transformer_bt import SwinUnet as ViT_seg
from config import get_config
import os
import argparse
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
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
if args.dataset == "Synapse":
    args.root_path = os.path.join(args.root_path, "train_npz")
config = get_config(args)
args = parser.parse_args()


if __name__ == '__main__':
    ds_train = brats_dataset('brats_dataset_processed')
    bs = 16
    base_lr = 0.01
    n_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl_train = DataLoader(ds_train, batch_size=bs)
    net = ViT_seg(get_config(args), img_size=256, num_classes=4)
    net = net.to(device)

    criterion = MSELoss()
    optimizer = Adam(lr=base_lr, params=net.parameters())
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs*len(dl_train), eta_min=0.00001)

    for epoch in range(n_epochs):
        net.train()
        losses = []
        for img1, img2, lbl in dl_train:
            img1 = img1.to(device)
            img2 = img2.to(device)
            optimizer.zero_grad()
            out1, out2 = net(img1, img2)

            # out1_np = out1.detach().cpu().numpy()
            # out2_np = out2.detach().cpu().numpy()

            out1_np_norm = (out1 - out1.mean(0)) / out1.std(0)
            out2_np_norm = (out2 - out2.mean(0)) / out2.std(0)

            cc = torch.matmul(out1_np_norm.T, out2_np_norm).to(device)

            gt = torch.eye(cc.shape[0]).to(device)

            loss = criterion(cc, gt)
            loss.backward()

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
    print("epoch loss : ", round(np.mean(losses), 4))

    torch.save(net, 'bt_pretrained_swin.pt')
