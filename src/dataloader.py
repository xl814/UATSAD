from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class AESegLoader(Dataset):
    def __init__(self, data, win_size, flag="train"):
        super().__init__()
        self.flag = flag
        self.win_size = win_size
        self.data = data
        # print(f"{flag} data shape: {self.data.shape}")


    def __len__(self):
        return self.data.shape[0] - self.win_size + 1

    def __getitem__(self, index):
        return np.float32(self.data[index: index + self.win_size])

class AEReconLoader(Dataset):
    def __init__(self, data, win_size, flag = "test"):
        super().__init__()
        self.win_size = win_size
        self.data = data

    def __len__(self):
        return self.data.shape[0] // self.win_size

    def __getitem__(self, index):
        return np.float32(self.data[index * self.win_size: (index + 1) * self.win_size])
