import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset

def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan

class PointCloudData(Dataset):
    """Some Information about MyDataset"""
    def __init__(self, dir):
        super(PointCloudData, self).__init__()
        self.dir = dir
        self.count = len(os.listdir(dir))

    def __getitem__(self, index):
        return torch.from_numpy(load_velo_scan(f"{self.dir}/{index:06d}.bin"))

    def __len__(self):
        return self.count
