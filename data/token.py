import os
import ast
import pandas as pd

import torch
from torch.utils.data import Dataset

from data.utils import get_split_df


class Genomics(Dataset):
    # Token representation of genomic bases.
    BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    def __init__(self, root, pre_train=True, train=True, ood=False):
        self.ood = ood
        filename = 'Pretrain' if pre_train else 'TransferOOD' if ood else 'TransferID'
        self.csv_root = os.path.join(root, 'genomics', f'genomics{filename}.csv')

        genomics_df = pd.read_csv(self.csv_root)
        genomics_df = genomics_df.sample(frac=1, random_state=42).reset_index(drop=True)
        genomics_df = genomics_df.iloc[:, 2:4]
        self.genomics_df = get_split_df(genomics_df, train, train_ratio=0.9)

    def __len__(self):
        return len(self.genomics_df)

    def __getitem__(self, index):
        row = self.genomics_df.iloc[index]
        label = row[0] - 10 if self.ood else row[0]
        seq   = row[1][2:252]
        tokens = []
        for base in seq:
            tokens.append(self.BASES[base])
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens, label


class Pfam(Dataset):
    def __init__(self, root, pre_train=True, train=True):
        super().__init__()
        filename = 'pfam_pretrain_train' if pre_train else 'pfam_transfer'
        self.csv_dir = os.path.join(root, 'pfam', f'{filename}.csv')
        self.pfam_df = pd.read_csv(self.csv_dir)
        if not pre_train:
            self.pfam_df = get_split_df(self.pfam_df, train, train_ratio=0.9)
        self.pfam_df = self.pfam_df.iloc[:, 1:]

        #vocab map
        self.amino_acid_map = {amino_acid:i for i, amino_acid in enumerate("XARNDCQEGHILKMFPSTWYVUOBZJ")}

        #labeling map
        if pre_train:
            pfam_pretrain = self.pfam_df
        else:
            pfam_pretrain = pd.read_csv(os.path.join(root, 'pfam', 'pfam_pretrain_train.csv'))

        clans = list(set(pfam_pretrain['clan']))
        clans.sort()
        self.label_map = {clan:i for i, clan in enumerate(clans)}

    def __len__(self):
        return len(self.pfam_df)

    def __getitem__(self, index):
        row = self.pfam_df.iloc[index]
        seq = row['primary']
        seq = (seq[:128]).ljust(128, 'X')
        tokens = []
        for amino_acid in seq:
            tokens.append(self.amino_acid_map[amino_acid])
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        clan = row['clan']
        label = self.label_map[clan]
        return token_tensor, label


class ProteinTransfer(Dataset):
    def __init__(self, root, dataset_name, train=True):
        filename = '_train' if train else '_valid'
        self.csv_dir = os.path.join(root, 'pfam', 'transfer', dataset_name+f'{filename}.csv')
        self.data = pd.read_csv(self.csv_dir)

        #vocab map
        self.amino_acid_map = {amino_acid:i for i, amino_acid in enumerate("XARNDCQEGHILKMFPSTWYVUOBZJ")}

    def __len__(self):
        return len(self.data)

    def __getrow__(self, index):
        row = self.data.iloc[index]
        seq = row['primary']
        seq = (seq[:128]).ljust(128, 'X')
        tokens = []
        for amino_acid in seq:
            tokens.append(self.amino_acid_map[amino_acid])
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        return token_tensor, row


class SCOP(ProteinTransfer):
    def __init__(self, root, train=True):
        super().__init__(root, 'remote_homology', train)

    def __getitem__(self, index):
        token_tensor, row = super().__getrow__(index)
        return token_tensor, row['fold_label']


class SecondaryStructure(ProteinTransfer):
    def __init__(self, root, train=True):
        super().__init__(root, 'secondary_structure', train)

    def __getitem__(self, index):
        token_tensor, row = super().__getrow__(index)
        secondary_struct_seq = (ast.literal_eval(row['ss3']))[:128]
        for _ in range(128 - len(secondary_struct_seq)):
            secondary_struct_seq.append(3)
        secondary_struct_seq = torch.tensor(secondary_struct_seq)
        return token_tensor, secondary_struct_seq


class Stability(ProteinTransfer):
    def __init__(self, root, train=True):
        super().__init__(root, 'stability', train)

    def __getitem__(self, index):
        token_tensor, row = super().__getrow__(index)
        score = np.float32(row['stability_score'][1:-1])
        return token_tensor, score


class Fluorescence(ProteinTransfer):
    def __init__(self, root, train=True):
        super().__init__(root, 'fluorescence', train)

    def __getitem__(self, index):
        token_tensor, row = super().__getrow__(index)
        # Convert log_fluorescence score from string to np float
        score = np.float32(row['log_fluorescence'][1:-1])
        return token_tensor, score


def get_dataset_config(dataset):
    if dataset in ['genomics', 'genomics-id', 'genomics-ood']:
        if dataset  == 'genomics':
            num_classes = 10
            batch_size  = 32
        elif dataset == 'genomics-id':
            num_classes = 10
            batch_size  = 64
        else:
            num_classes = 60
            batch_size  = 32
        input_shape = ((4,), 250)
        patch_size  = (1,)
    ## token - proteins
    elif dataset in ['pfam', 'scop', 'secondary-structure', 'stability', 'fluorescence']:
        if dataset == 'pfam':
            num_classes = 623
        elif dataset == 'scop':
            num_classes = 1195
        elif dataset == 'secondary-structure':
            num_classes = 4
        else:
            num_classes = 0
        input_shape = ((26,), 128) #vocab size=26
        patch_size  = (1,)
        batch_size  = 128
    return dict(num_classes=num_classes,
                input_shape=input_shape,
                patch_size=patch_size,
                batch_size=batch_size)


def get_pretrain_dataset(dataset, datadir):
    if dataset == 'genomics':
        train = Genomics(datadir, pre_train=True, train=True)
    elif dataset == 'pfam':
        train = Pfam(datadir, pre_train=True, train=True)
    else:
        raise Exception(f'No pretrain dataset: {dataset}')
    return train


def get_transfer_dataset(dataset, datadir):
    ## genomics
    if dataset == 'genomics-id':
        val  = Genomics(datadir, pre_train=False, train=True,  ood=False)
        test = Genomics(datadir, pre_train=False, train=False, ood=False)
    elif dataset == 'genomics-ood':
        val  = Genomics(datadir, pre_train=False, train=True,  ood=True)
        test = Genomics(datadir, pre_train=False, train=False, ood=True)
    ## proteins
    elif dataset == 'pfam':
        val  = Pfam(datadir, pre_train=False, train=True)
        test = Pfam(datadir, pre_train=False, train=False)
    elif dataset == 'scop':
        val  = SCOP(datadir, train=True)
        test = SCOP(datadir, train=False)
    elif dataset == 'secondary-structure':
        val  = SecondaryStructure(datadir, train=True)
        test = SecondaryStructure(datadir, train=False)
    elif dataset == 'stability':
        val  = Stability(datadir, train=True)
        test = Stability(datadir, train=False)
    elif dataset == 'fluorescence':
        val  = Fluorescence(datadir, train=True)
        test = Fluorescence(datadir, train=False)
    else:
        raise Exception(f'No transfer dataset: {dataset}')
    return val, test

