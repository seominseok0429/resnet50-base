from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image, ImageFilter
import glob

class Loader(Dataset):
    """
    is_train - 0 train, 1 val
    """
    def __init__(self, is_train=0, path='', transform=None):
        self.is_train = is_train
        self.transform = transform

        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        pass

if __name__ == '__main__':
    loader = Loader()
    print(len(loader))
