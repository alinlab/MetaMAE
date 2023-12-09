import os
from glob import glob
from collections import defaultdict

from PIL import Image, ImageFile

from scipy.io import loadmat

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import torchvision.datasets as VD
import torchvision.transforms as T

from data.utils import get_split_df


class CUB(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.train = train
        self.root = root
        self.transform = transform
        self.paths, self.labels = self.load_images()

    def load_images(self):
        # load id to image path information
        image_info_path = os.path.join(self.root, 'CUB_200_2011', 'images.txt')
        with open(image_info_path, 'r') as f:
            image_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        image_info = dict(image_info)

        # load image to label information
        label_info_path = os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt')
        with open(label_info_path, 'r') as f:
            label_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        label_info = dict(label_info)

        # load train test split
        train_test_info_path = os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')
        with open(train_test_info_path, 'r') as f:
            train_test_info = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        train_test_info = dict(train_test_info)

        all_paths, all_labels = [], []
        for index, image_path in image_info.items():
            label = label_info[index]
            split = int(train_test_info[index])
            if self.train:
                if split == 1:
                    all_paths.append(image_path)
                    all_labels.append(label)
            else:
                if split == 0:
                    all_paths.append(image_path)
                    all_labels.append(label)
        return all_paths, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.root, 'CUB_200_2011', 'images', self.paths[index])
        label = int(self.labels[index]) - 1
        image = Image.open(path).convert(mode='RGB')
        image = self.transform(image)

        return image, label


class VGGFlower(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.train = train
        self.transform = transform
        self.root = os.path.join(root, 'flowers102')
        self.paths, self.labels = self.load_images()

    def load_images(self):
        rs = np.random.RandomState(42)
        imagelabels_path = os.path.join(self.root, 'imagelabels.mat')
        with open(imagelabels_path, 'rb') as f:
            labels = loadmat(f)['labels'][0]

        all_filepaths = defaultdict(list)
        for i, label in enumerate(labels):
            all_filepaths[label].append(os.path.join(self.root, 'jpg', 'image_{:05d}.jpg'.format(i + 1)))
        # train test split
        split_filepaths, split_labels = [], []
        for label, paths in all_filepaths.items():
            num = len(paths)
            paths = np.array(paths)
            indexer = np.arange(num)
            rs.shuffle(indexer)
            paths = paths[indexer].tolist()
            paths = paths[:int(0.8 * num)] if self.train else  paths[int(0.8 * num):]
            labels = [label] * len(paths)
            split_filepaths.extend(paths)
            split_labels.extend(labels)

        return split_filepaths, split_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path, label = self.paths[index], int(self.labels[index]) - 1
        image = Image.open(path).convert(mode='RGB')
        image = self.transform(image)
        return image, label


class DTD(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.transform = transform
        self.root = os.path.join(root, 'imagenet')
        self.paths, self.labels = self.load_images()

    def __len__(self):
        return len(self.paths)

    def load_images(self):
        if self.train:
            train_info_path = os.path.join(self.root, 'dtd', 'labels', 'train1.txt')
            with open(train_info_path, 'r') as f:
                train_info = [line.split('\n')[0] for line in f.readlines()]

            val_info_path = os.path.join(self.root, 'dtd', 'labels', 'val1.txt')
            with open(val_info_path, 'r') as f:
                val_info = [line.split('\n')[0] for line in f.readlines()]

            split_info = train_info + val_info
        else:
            test_info_path = os.path.join(self.root, 'dtd', 'labels', 'test1.txt')
            with open(test_info_path, 'r') as f:
                split_info = [line.split('\n')[0] for line in f.readlines()]

        # pull out categoires from paths
        categories = []
        for row in split_info:
            image_path = row
            category = image_path.split('/')[0]
            categories.append(category)
        categories = sorted(list(set(categories)))

        all_paths, all_labels = [], []
        for row in split_info:
            image_path = row
            category = image_path.split('/')[0]
            label = categories.index(category)
            all_paths.append(os.path.join(self.root, 'dtd', 'images', image_path))
            all_labels.append(label)

        return all_paths, all_labels

    def __getitem__(self, index):
        path, label = self.paths[index], self.labels[index]
        image = Image.open(path).convert(mode='RGB')
        image = self.transform(image)
        return image, label


class TrafficSign(Dataset):
    def __init__(self, root, train=True):
        self.train = train
        self.root = os.path.join(root, 'imagenet', 'traffic_sign')
        self.transform = T.Compose([T.Resize((32, 32)),
                                    T.CenterCrop((32, 32)),
                                    T.ToTensor()])
        self.paths, self.labels = self.load_images()

    def load_images(self):
        rs = np.random.RandomState(42)
        all_filepaths, all_labels = [], []
        for class_i in range(43):
            class_dir_i = os.path.join(self.root, 'GTSRB', 'Final_Training', 'Images', '{:05d}'.format(class_i))
            image_paths = glob(os.path.join(class_dir_i, '*.ppm'))
            image_paths = np.array(image_paths)
            num = len(image_paths)
            indexer = np.arange(num)
            rs.shuffle(indexer)
            image_paths = image_paths[indexer].tolist()
            if self.train:
                image_paths = image_paths[:int(0.8 * num)]
            else:
                image_paths = image_paths[int(0.8 * num):]
            labels = [class_i] * len(image_paths)
            all_filepaths.extend(image_paths)
            all_labels.extend(labels)

        return all_filepaths, all_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = Image.open(path).convert(mode='RGB')
        image = self.transform(image)
        return image, label


class Aircraft(Dataset):
    def __init__(self, root, train=True):
        super().__init__()
        self.train = train
        self.root = os.path.join(root, 'imagenet', 'fgvc-aircraft-2013b')
        self.transform = T.Compose([T.Resize((32,32)),
                                    T.CenterCrop((32, 32)),
                                    T.ToTensor()])
        paths, bboxes, labels = self.load_images()
        self.paths = paths
        self.bboxes = bboxes
        self.labels = labels

    def load_images(self):
        split = 'trainval' if self.train else 'test'
        variant_path = os.path.join(self.root, 'data', f'images_variant_{split}.txt')
        with open(variant_path, 'r') as f:
            names_to_variants = [line.split('\n')[0].split(' ', 1) for line in f.readlines()]
        names_to_variants = dict(names_to_variants)
        variants_to_names = defaultdict(list)
        for name, variant in names_to_variants.items():
            variants_to_names[variant].append(name)

        names_to_bboxes = self.get_bounding_boxes()

        variants = sorted(list(set(variants_to_names.keys())))
        split_files, split_labels, split_bboxes = [], [], []

        for variant_id, variant in enumerate(variants):
            class_files = [
                os.path.join(self.root, 'data', 'images', f'{filename}.jpg')
                for filename in sorted(variants_to_names[variant])
            ]
            bboxes = [names_to_bboxes[name] for name in sorted(variants_to_names[variant])]
            labels = list([variant_id] * len(class_files))

            split_files += class_files
            split_labels += labels
            split_bboxes += bboxes

        return split_files, split_bboxes, split_labels

    def get_bounding_boxes(self):
        bboxes_path = os.path.join(self.root, 'data', 'images_box.txt')
        with open(bboxes_path, 'r') as f:
            names_to_bboxes = [line.split('\n')[0].split(' ') for line in f.readlines()]
            names_to_bboxes = dict(
                (name, list(map(int, (xmin, ymin, xmax, ymax)))) for name, xmin, ymin, xmax, ymax in names_to_bboxes
            )

        return names_to_bboxes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        bbox = tuple(self.bboxes[index])
        label = self.labels[index]

        image = Image.open(path).convert(mode='RGB')
        image = image.crop(bbox)
        image = self.transform(image)

        return image.float(), label


class WaferMap(Dataset):
    FAILURE_MAP = {'unlabeled': 0,
                   'none': 0,
                   'random': 1,
                   'donut': 2,
                   'scratch': 3,
                   'center': 4,
                   'loc': 5,
                   'edge-loc': 6,
                   'edge-ring': 7,
                   'near-full': 8}
    def __init__(self, root, pre_train=True, train=True):
        self.root = os.path.join(root, 'wafer_map')
        self.transforms = T.Compose([T.Resize((32, 32)),
                                     T.ToTensor()])

        pkl_file = 'unlabeled.pkl' if pre_train else 'labeled.pkl'
        self.data = pd.read_pickle(os.path.join(self.root, pkl_file))
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.data = get_split_df(self.data, train, train_ratio=0.9)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img, label = row['pixels'], self.FAILURE_MAP[row['failureType']]
        img = img.astype('uint8')
        img = Image.fromarray(img).convert('RGB')
        img = self.transforms(img)
        return img, label


def get_dataset_config(dataset):
    ## imagenet32
    if dataset in ['imagenet32', 'cifar10', 'cub', 'vggflower', 'dtd', 'traffic-sign', 'aircraft']:
        if dataset == 'imagenet32':
            num_classes = 1000
        elif dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cub':
            num_classes = 200
        elif dataset == 'vggflower':
            num_classes = 102
        elif dataset == 'dtd':
            num_classes = 47
        elif dataset == 'traffic-sign':
            num_classes = 43
        input_shape = (3, 32, 32)
        patch_size = (4, 4)
        batch_size = 64
    ## wafer-map
    elif dataset == 'wafer-map':
        num_classes = 9
        input_shape = (3, 32, 32)
        patch_size = (4, 4)
        batch_size = 128
    return dict(num_classes=num_classes,
                input_shape=input_shape,
                patch_size=patch_size,
                batch_size=batch_size)


def get_pretrain_dataset(dataset, datadir):
    if dataset == 'imagenet32':
        transform = T.Compose([T.Resize((32, 32)),
                               T.CenterCrop((32, 32)),
                               T.ToTensor()])
        train = VD.ImageNet(datadir, 'train', transform=transform)
    elif dataset == 'wafer-map':
        train = WaferMap(datadir, pre_train=True, train=True)
    else:
        raise Exception(f'No pretrain dataset: {dataset}')
    return train


def get_transfer_dataset(dataset, datadir):
    ## imagenet32
    img_transform = T.Compose([T.Resize((32, 32)),
                                 T.CenterCrop((32, 32)),
                                 T.ToTensor()])
    if dataset == 'cifar10':
        val  = VD.CIFAR10(datadir, train=True,  transform=img_transform)
        test = VD.CIFAR10(datadir, train=False, transform=img_transform)
    elif dataset == 'cub':
        val  = CUB(datadir, train=True,  transform=img_transform)
        test = CUB(datadir, train=False, transform=img_transform)
    elif dataset == 'vggflower':
        val  = VGGFlower(datadir, train=True,  transform=img_transform)
        test = VGGFlower(datadir, train=False, transform=img_transform)
    elif dataset == 'dtd':
        val  = DTD(datadir, train=True,  transform=img_transform)
        test = DTD(datadir, train=False, transform=img_transform)
    ## wafer-map
    elif dataset == 'wafer-map':
        val  = WaferMap(datadir, pre_train=False, train=True)
        test = WaferMap(datadir, pre_train=False, train=False)
    else:
        raise Exception(f'No transfer dataset: {dataset}')
    return val, test

