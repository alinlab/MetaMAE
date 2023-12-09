import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class PAMAP2(Dataset):
    TRAIN_EXAMPLES_PER_EPOCH = 50000  # examples are generated stochastically
    VAL_EXAMPLES_PER_EPOCH = 10000
    MEASUREMENTS_PER_EXAMPLE = 320
    ACTIVITY_LABELS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

    def __init__(self, root, train=True):
        super().__init__()
        self.root = os.path.join(root, 'sensor', 'pamap2')
        self.mode = 'train' if train else 'val'

        self.data = self.load_data()
        self.samples = self.get_candidates(self.data)

    def load_data(self):
        subject_data = []
        nums = [1, 2, 3, 4, 7, 8, 9] if self.mode == 'train' else [5, 6]
        for subject_filename in [f'subject10{num}.dat' for num in nums]:
            columns = ['timestamp', 'activity_id', 'heart_rate']
            for part in ['hand', 'chest', 'ankle']:
                for i in range(17):
                    columns.append(part + str(i))
            subj_path = os.path.join(self.root, 'Protocol', subject_filename)
            subj_path_cache = subj_path + '.p'
            if os.path.isfile(subj_path_cache):
                df = pd.read_pickle(subj_path_cache)
            else:
                df = pd.read_csv(subj_path, names=columns, sep=' ')
                df = df.interpolate()
                df.to_pickle(subj_path_cache)
            subject_data.append(df)
            print(f'load done: {subject_filename}', end='\r')

        return subject_data

    def get_candidates(self, data):
        samples = []
        for df in data:
            for activity_id in range(len(self.ACTIVITY_LABELS)):
                activity_data = df[df['activity_id'] == self.ACTIVITY_LABELS[activity_id]].to_numpy()
                if len(activity_data) > self.MEASUREMENTS_PER_EXAMPLE:
                    samples.append((activity_data, activity_id))

        return samples

    def __len__(self):
        return self.TRAIN_EXAMPLES_PER_EPOCH if self.mode == 'train' else self.VAL_EXAMPLES_PER_EPOCH

    def __getitem__(self, index):
        sample_id = np.random.randint(len(self.samples))
        activity_data, activity_id = self.samples[sample_id]
        start_idx = np.random.randint(len(activity_data) - self.MEASUREMENTS_PER_EXAMPLE)
        x = activity_data[start_idx:start_idx + self.MEASUREMENTS_PER_EXAMPLE, 2:].T
        x = torch.tensor(x, dtype=torch.float32)
        return x, activity_id


def get_dataset_config(dataset):
    if dataset == 'pamap2':
        num_classes = 12
        input_shape = (52, 320)
        patch_size  = (5,)
        batch_size  = 256
    return dict(num_classes=num_classes,
                input_shape=input_shape,
                patch_size=patch_size,
                batch_size=batch_size)


def get_pretrain_dataset(dataset, datadir):
    if dataset == 'pamap2':
        train =  PAMAP2(datadir, train=True)
    else:
        raise Exception(f'No pretrain dataset: {dataset}')
    return train


def get_transfer_dataset(dataset, datadir):
    if dataset == 'pamap2':
        val  = PAMAP2(datadir, train=True)
        test = PAMAP2(datadir, train=False)
    else:
        raise Exception(f'No transfer dataset: {dataset}')
    return val, test

