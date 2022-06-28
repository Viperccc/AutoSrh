import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def train_val_test_split(file, test_ratio=0.1, val_ratio=0.2, seed=12):
    data_df = pd.read_csv(file)
    train, test = train_test_split(data_df, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(train, test_size=val_ratio / (1 - test_ratio), random_state=seed)
    return train, val, test, data_df


class CFDataset(Dataset):
    def __init__(self, data_df):
        self.x = torch.LongTensor(np.array(data_df.drop(columns=['rating'])))
        self.y = torch.Tensor(np.array(data_df['rating']))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
