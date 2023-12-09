import os

import pandas as pd

from torch.utils.data import Dataset

import torchvision.transforms as T

from data.utils import get_split_df


class EuroSAT(Dataset):
    MEAN = [1354.3003, 1117.7579, 1042.2800, 947.6443,  1199.6334, 2001.9829, 2372.5579,
            2299.6663, 731.0175,  12.0956,   1822.4083, 1119.5759, 2598.4456]
    STD  = [244.0469,  323.4128,  385.0928,  584.1638,  566.0543,  858.5753,  1083.6704,
            1103.0342, 402.9594,  4.7207,    1002.4071, 759.6080,  1228.4104]
    def __init__(self, root, pre_train=True, train=True):
        super().__init__()
        filename = 'pretrain' if pre_train else 'transfer'
        self.pkl_dir = os.path.join(root, 'eurosat_all', f'{filename}.pkl')

        eurosat_df = pd.read_pickle(self.pkl_dir)
        self.eurosat = get_split_df(eurosat_df, train, train_ratio=0.9)

        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=self.MEAN, std=self.STD)])

    def __len__(self):
        return len(self.eurosat)

    def __getitem__(self, index):
        row = self.eurosat.iloc[index]
        label = row['label']
        img = row['sentinel2']

        return self.transform(img), label


def get_dataset_config(dataset):
    if dataset == 'eurosat':
        num_classes = 10
        input_shape = (13, 64, 64)
        patch_size  = (8, 8)
        batch_size  = 64
    return dict(num_classes=num_classes,
                input_shape=input_shape,
                patch_size=patch_size,
                batch_size=batch_size)


def get_pretrain_dataset(dataset, datadir):
    if dataset == 'eurosat':
        train = EuroSAT(datadir, pre_train=True,  train=True)
    else:
        raise Exception(f'No pretrain dataset: {dataset}')
    return train


def get_transfer_dataset(dataset, datadir):
    if dataset == 'eurosat':
        val  = EuroSAT(datadir, pre_train=False, train=True)
        test = EuroSAT(datadir, pre_train=False, train=False)
    else:
        raise Exception(f'No transfer dataset: {dataset}')
    return val, test

