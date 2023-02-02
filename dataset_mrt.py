import torch
from torch.utils.data import Dataset

import numpy as np

class mrtDataset(Dataset):
    def __init__(self, items, get_data, train=True):
        self.items = items
        self.get_data = get_data
        self.train = train
        print('Total number of files in loader: ', len(self.items))

    def __getitem__(self, index):
        if self.train:
            item = self.items[index//125]
            data = self.get_data(item[0], item[1])
            return data[index%125, :, :]
        else:
            item = self.items[index//20]
            data = self.get_data(item[0], item[1])
            return data[index%20, :, :]

    def __len__(self):
        if self.train:
            return (len(self.items)*125)
        else:
            return (len(self.items)*20)