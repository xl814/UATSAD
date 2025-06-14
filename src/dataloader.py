from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class VAESegLoader(Dataset):
    def __init__(self, data, win_size, flag="train", labels=None, heteroscedastic=None):
        super().__init__()
        self.flag = flag
        self.win_size = win_size
        self.data = data

        self.labels = labels
        self.heteroscedastic = heteroscedastic
        # print(f"{flag} data shape: {self.data.shape}")
        # split the data by the window size
        # ts_to_window = lambda ts: [ts[i: i + win_size] for i in range(0, len(ts) - win_size + 1)]
        # self.train = np.array(ts_to_window(self.train))
        # self.test = np.array(ts_to_window(self.test))
        # self.val = np.array(ts_to_window(self.val))


    def __len__(self):
        return self.data.shape[0] - self.win_size + 1

    def __getitem__(self, index):
        if self.labels is not None: 
            return np.float32(self.data[index: index + self.win_size]), np.float32(self.labels[index: index + self.win_size])
        elif self.heteroscedastic is not None:
            return np.float32(self.data[index: index + self.win_size]), np.float32(np.ones(self.win_size))
        return np.float32(self.data[index: index + self.win_size])

class VAEReconLoader(Dataset):
    def __init__(self, data, win_size, flag = "test"):
        super().__init__()
        self.win_size = win_size
        self.data = data
        # print(f"{flag} data shape: {self.data.shape}")
        # split the data by the window size
        # ts_to_window = lambda ts: [ts[i: i + win_size] for i in range(0, len(ts) - win_size + 1)]
        # self.train = np.array(ts_to_window(self.train))
        # self.test = np.array(ts_to_window(self.test))
        # self.val = np.array(ts_to_window(self.val))


    def __len__(self):
        return self.data.shape[0] // self.win_size

    def __getitem__(self, index):
        return np.float32(self.data[index * self.win_size: (index + 1) * self.win_size])


class LSTMDataset(Dataset):
    """
    Args:
    - `data`: np.array, shape=(n_samples, seq_size, win_size, latent_dim)
    """
    def __init__(self, data, flag="train"):
        super().__init__()
        self.data = data
        self.seq_size = data.shape[1]
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return np.float32(self.data[index, :self.seq_size - 1]), np.float32(self.data[index, 1:])
        # return np.float32(self.data[index: index + self.win_size]), np.float32(self.data[index + self.win_size - 1])

class LSTMDataset2(Dataset):
    """
    Args:
    - `data`: np.array, shape=(n_samples, seq_size, win_size, latent_dim)
    """
    def __init__(self, data, flag="train"):
        super().__init__()
        self.data = data
        self.seq_size = data.shape[1]
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return np.float32(self.data[index, :self.seq_size - 1]), np.float32(self.data[index, 1:])
        # return np.float32(self.data[index: index + self.win_size]), np.float32(self.data[index + self.win_size - 1])

class EmbeddingDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return np.float32(self.data[index])