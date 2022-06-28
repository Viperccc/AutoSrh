import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pickle
import os


def train_val_test_split(data_df, test_ratio=0.1, val_ratio=0.2, seed=12):
    train, test = train_test_split(data_df, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(train, test_size=val_ratio / (1 - test_ratio), random_state=seed)
    return train, val, test


class CF_Dataset(Dataset):
    def __init__(self, data_df):
        self.x = torch.LongTensor(np.array(data_df.drop(columns=['rating'])))
        self.y = torch.Tensor(np.array(data_df['rating']))

    def __getitem__(self, idx):
        return self.x[idx], torch.ones_like(self.x[idx], dtype=torch.float32), self.y[idx]

    def __len__(self):
        return len(self.x)


class CTR_Dataset(Dataset):
    def __init__(self, id_df, value_df):
        self.x_id = torch.LongTensor(np.array(id_df.drop(columns=['rating'])))
        self.y = torch.Tensor(np.array(id_df['rating']))
        self.x_value = torch.Tensor(np.array(value_df))

    def __getitem__(self, idx):
        return self.x_id[idx], self.x_value[idx], self.y[idx]

    def __len__(self):
        return self.x_id.shape[0]


def get_ctr_dataset(data_path):
    if os.path.isfile(f"{data_path}"):
        print("The dataset has been processed. Reading the cache...")
        with open(f"{data_path}", 'rb') as handle:
            result = pickle.load(handle)
        return result
    else:
        raise FileNotFoundError("The preprocessed dataset is not found.")


def get_cf_dataset(data_path, filter_item=True, test_ratio=0.1, val_ratio=0.2, seed=12):
    if os.path.isfile(f"{data_path}.pickle"):
        print("The dataset has been processed. Reading the cache...")
        with open(f"{data_path}.pickle", 'rb') as handle:
            result = pickle.load(handle)
        return result
    print("Preprocessing the dataset...")
    rating_df = pd.read_csv(data_path)
    if filter_item:
        rating_df = rating_df.groupby('movieId').filter(lambda x: len(x) >= 5)
    rating_df['userId'] = rating_df['userId'].astype('category').cat.codes
    rating_df['movieId'] = rating_df['movieId'].astype('category').cat.codes
    rating_df = rating_df.drop(["timestamp"], axis=1)
    rating_df['movieId'] += rating_df['userId'].max() + 1
    num_features = rating_df['movieId'].max() + 1
    print("Remapping feature ids according to the frequency...")
    freq_list = rating_df.groupby('userId').size().to_list() + rating_df.groupby('movieId').size().to_list()
    freq_list_sorted = sorted(range(len(freq_list)), key=lambda k: freq_list[k])[::-1]
    freq_map = {k: v for v, k in enumerate(freq_list_sorted)}
    rating_df['userId'] = rating_df['userId'].map(freq_map)
    rating_df['movieId'] = rating_df['movieId'].map(freq_map)
    print("Splitting...")
    train_df, val_df, test_df = train_val_test_split(rating_df, test_ratio, val_ratio, seed)
    print("The dataset has been preprocessed.")
    result = [CF_Dataset(train_df), CF_Dataset(val_df), CF_Dataset(test_df), num_features]
    with open(f"{data_path}.pickle", 'wb') as handle:
        pickle.dump(result, handle)
        print("File saved.")
    return result
