import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy
from torchvision import transforms
import csv
import pandas as pd
import matplotlib.pyplot as plt

def save_table_as_image(data, filename):
    # Convert the list of lists into a Pandas DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])

    # Create a figure and axis without frame
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')  # Hide the axis

    # Render the DataFrame as a table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    # Adjust font size and cell padding
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save the table as an image
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.5)
    plt.close()  # Close the plot to avoid displaying it


class BEVMaskCloudData(Dataset):
    """Some Information about MyDataset"""
    def __init__(self, dir):
        super(BEVMaskCloudData, self).__init__()
        self.dir = dir
        self.count = len(os.listdir(dir))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        return self.transform(Image.open(f"{self.dir}/{index:06d}.png"))

    def __len__(self):
        return self.count


def analyze_image(im, vote_percent=0.5, vote_count=4, block_sizes=[1, 2, 4, 8, 16, 32, 64, 128]):
    im = im.cuda().squeeze()

    win_counts = []
    for block_size in block_sizes:
        new_height = im.shape[0] // block_size
        new_width = im.shape[1] // block_size

        r_im = im.view(new_height, block_size, new_width, block_size, -1)
        r_im = r_im.permute(0, 2, 1, 3, 4).detach().clone().contiguous().view(new_height, new_width, block_size * block_size)

        r_im = torch.count_nonzero(r_im, dim=-1).to(torch.float)
        block_winners = torch.logical_or(r_im >= vote_count, r_im / (block_size * block_size) >= vote_percent)
        win_counts.append(block_winners.count_nonzero().item())
    
    return torch.tensor(win_counts)

def make_table(dirs, hp, t):
    win_counts = None
    table = [[".", ".", 1, 2, 4, 8, 16, 32, 64, 128]]
    for ci, c_dir in enumerate(dirs):
        ds = BEVMaskCloudData(c_dir)
        for i in range(len(ds)):
            wc = analyze_image(ds[i])
            if win_counts is None: win_counts = wc
            else: win_counts += wc
            # if i % 100 == 0: print("At Image", i, "in", "Count", ci)
        win_counts = win_counts.to(torch.float) / len(ds)
        table.append([t, hp[ci], *win_counts.tolist()])
    print(t.lower(), "= ", end='')
    print(table)
    print("\n\n")


if __name__ == "__main__":
    src_dir = "detectron2/datasets/bv_kitti/velodyne_3d_points"
    src_train_dir = f"{src_dir}/training/velodyne/"

    im_dst_dir = "detectron2/datasets/bv_kitti/image/"
    sp_dst_dir = "detectron2/datasets/bv_kitti/sparse/"

    count_hp = [1, 2, 3, 4]
    density_hp = [0.01, 0.02, 0.04, 0.08]
    height_hp = [0.0, 0.2, 0.4, 0.6]
    count_dirs = [f'{sp_dst_dir}/count_{hp}' for hp in count_hp]
    density_dirs = [f'{sp_dst_dir}/density_{hp}' for hp in density_hp]
    height_dirs = [f'{sp_dst_dir}/height_{hp}' for hp in height_hp]

    make_table(count_dirs, count_hp, "Count")
    make_table(density_dirs, density_hp, "Density")
    make_table(height_dirs, height_hp, "Height")
