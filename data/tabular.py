import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from data.utils import get_split_df


class HIGGS(Dataset):
    def __init__(self, root, pre_train=True, train=True):
        super().__init__()
        filename = 'higgsPretrain' if pre_train else 'higgsTransfer'
        self.csv_dir = os.path.join(root, 'particle_physics', f'{filename}.csv')

        higgs_df = pd.read_csv(self.csv_dir)
        higgs_df = higgs_df.sample(frac=1, random_state=42).reset_index(drop=True)
        higgs_df = higgs_df.iloc[:, 1:]
        self.higgs_df = get_split_df(higgs_df, train, train_ratio=0.9)

    def __len__(self):
        return len(self.higgs_df)

    def __getitem__(self, index):
        row = self.higgs_df.iloc[index]
        label = int(row[0])

        features = torch.tensor(row[1:])
        features = features.view(1, 28)
        return features.float(), np.array([label])


def get_dataset_config(dataset):
    if dataset == 'higgs':
        num_classes = 2
        input_shape = (1, 28)
        patch_size  = (1,)
        batch_size  = 256
    return dict(num_classes=num_classes,
                input_shape=input_shape,
                patch_size=patch_size,
                batch_size=batch_size)


def get_pretrain_dataset(dataset, datadir):
    if dataset == 'higgs':
        train = HIGGS(datadir, pre_train=True,  train=True)
    else:
        raise Exception(f'No pretrain dataset: {dataset}')
    return train


def get_transfer_dataset(dataset, datadir):
    if dataset == 'higgs':
        val  = HIGGS(datadir, pre_train=False, train=True)
        test = HIGGS(datadir, pre_train=False, train=False)
    else:
        raise Exception(f'No transfer dataset: {dataset}')
    return val, test

