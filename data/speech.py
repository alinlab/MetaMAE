import os
import random
from glob import glob
from collections import defaultdict

import librosa

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import torchaudio
import torchaudio.datasets as AD

import torchvision.transforms as T


class LibriSpeech(Dataset):
    MAX_LENGTH = 150526

    ALL_TRAIN_NUM_CLASSES = 2338
    DEV_CLEAN_NUM_CLASSES = 40

    mean = [-22.924]
    std  = [12.587]

    def __init__(self, root, download=True, pre_train=True, train=True):
        super().__init__()
        self.root = os.path.join(root, 'speech')

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if pre_train:
            self.dataset1 = AD.LIBRISPEECH(self.root, url='train-clean-100', download=download, folder_in_archive='LibriSpeech')
            self.dataset2 = AD.LIBRISPEECH(self.root, url='train-clean-360', download=download, folder_in_archive='LibriSpeech')
            self.dataset3 = AD.LIBRISPEECH(self.root, url='train-other-500', download=download, folder_in_archive='LibriSpeech')
        else:
            self.dataset = AD.LIBRISPEECH(self.root, url='dev-clean', download=download, folder_in_archive='LibriSpeech')

        self.pre_train = pre_train
        self.all_speaker_ids = self.get_speaker_ids()
        unique_speaker_ids = sorted(list(set(self.all_speaker_ids)))
        num_classes = self.ALL_TRAIN_NUM_CLASSES if pre_train else self.DEV_CLEAN_NUM_CLASSES
        assert num_classes == len(unique_speaker_ids)
        self.speaker_id_map = dict(zip(unique_speaker_ids, range(num_classes)))

        if not self.pre_train:
            self.indices = self.train_test_split(self.all_speaker_ids, train=train)

    def get_speaker_ids(self):
        if self.pre_train:
            speaker_ids_1 = self._get_speaker_ids(self.dataset1)
            speaker_ids_2 = self._get_speaker_ids(self.dataset2)
            speaker_ids_3 = self._get_speaker_ids(self.dataset3)
            return np.concatenate([speaker_ids_1, speaker_ids_2, speaker_ids_3])
        else:
            return self._get_speaker_ids(self.dataset)

    def _get_speaker_ids(self, dataset):
        speaker_ids = []
        for i in range(len(dataset)):
            fileid = dataset._walker[i]
            speaker_id = self.load_librispeech_speaker_id(
                fileid,
                dataset._path,
                dataset._ext_audio,
                dataset._ext_txt,
            )
            speaker_ids.append(speaker_id)
        return np.array(speaker_ids)

    def train_test_split(self, speaker_ids, train=True):
        rs = np.random.RandomState(42)  # fix seed so reproducible splitting

        unique_speaker_ids = sorted(set(speaker_ids))
        unique_speaker_ids = np.array(unique_speaker_ids)

        # train test split to ensure the 80/20 splits
        train_indices, test_indices = [], []
        for speaker_id in unique_speaker_ids:
            speaker_indices = np.where(speaker_ids == speaker_id)[0]
            size = len(speaker_indices)
            rs.shuffle(speaker_indices)
            train_size = int(0.8 * size)
            train_indices.extend(speaker_indices[:train_size].tolist())
            test_indices.extend(speaker_indices[train_size:].tolist())

        return train_indices if train else test_indices

    def load_librispeech_speaker_id(self, fileid, path, ext_audio, ext_txt):
        speaker_id, _, _ = fileid.split('-')
        return int(speaker_id)

    def __getitem__(self, index):
        if self.pre_train:
            if index >= (len(self.dataset1) + len(self.dataset2)):
                wavform, sample_rate, _, speaker_id, _, _ = self.dataset3.__getitem__(index - len(self.dataset1) - len(self.dataset2))
            elif index >= len(self.dataset1):
                wavform, sample_rate, _, speaker_id, _, _ = self.dataset2.__getitem__(index - len(self.dataset1))
            else:
                wavform, sample_rate, _, speaker_id, _, _ = self.dataset1.__getitem__(index)
        else:
            wavform, sample_rate, _, speaker_id, _, _ = self.dataset.__getitem__(self.indices[index])

        speaker_id = self.speaker_id_map[speaker_id]
        wavform = np.asarray(wavform[0])

        if len(wavform) > self.MAX_LENGTH:
            flip = (bool(random.getrandbits(1)) if self.pre_train else True)
            padded = (wavform[:self.MAX_LENGTH] if flip else wavform[-self.MAX_LENGTH:])
        else:
            padded = np.zeros(self.MAX_LENGTH)
            padded[:len(wavform)] = wavform

        spectrum = librosa.feature.melspectrogram(
            y=padded,
            sr=sample_rate,
            hop_length=672,
            n_mels=224,
        )

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        normalize = T.Normalize(self.mean, self.std)
        spectrum = normalize(spectrum)

        return spectrum, speaker_id

    def __len__(self):
        if self.pre_train:
            return len(self.dataset1) + len(self.dataset2) + len(self.dataset3)
        else:
            return len(self.indices)


class AudioMNIST(Dataset):
    AUDIOMNIST_TRAIN_SPK = [28, 56, 7,  19, 35, 1,  6,  16, 23, 34, 46, 53,
                            36, 57, 9,  24, 37, 2,  8,  17, 29, 39, 48, 54,
                            43, 58, 14, 25, 38, 3,  10, 20, 30, 40, 49, 55]
    AUDIOMNIST_VAL_SPK   = [12, 47, 59, 15, 27, 41, 4,  11, 21, 31, 44, 50]
    AUDIOMNIST_TEST_SPK  = [26, 52, 60, 18, 32, 42, 5,  13, 22, 33, 45, 51]
    MAX_LENGTH = 150526

    def __init__(self, root, train=True):
        super().__init__()
        self.root = os.path.join(root, 'speech', 'AudioMNIST')
        self.train = train

        speakers = self.AUDIOMNIST_TRAIN_SPK + self.AUDIOMNIST_VAL_SPK if train else self.AUDIOMNIST_TEST_SPK
        self.wav_paths = []
        for spk in speakers:
            spk_paths = glob(os.path.join(self.root, 'data', '{:02}'.format(spk), '*.wav'))
            self.wav_paths.extend(spk_paths)

        self.transform = T.Normalize([-90.293], [11.799])

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        label, _, _ = wav_path.rstrip('.wav').split('/')[-1].split('_')

        wavform, sample_rate = torchaudio.load(wav_path)
        wavform = wavform[0].numpy()

        if len(wavform) > self.MAX_LENGTH:
            # randomly pick which side to chop off (fix if validation)
            flip = (bool(random.getrandbits(1)) if self.train else True)
            padded = (wavform[:self.MAX_LENGTH] if flip else wavform[-self.MAX_LENGTH:])
        else:
            padded = np.zeros(self.MAX_LENGTH)
            padded[:len(wavform)] = wavform  # pad w/ silence

        spectrum = librosa.feature.melspectrogram(
            y=padded,
            sr=sample_rate,
            hop_length=672,
            n_mels=224,
        )
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)
        spectrum = self.transform(spectrum)

        return spectrum, int(label)


class FluentSpeechCommand(Dataset):
    FLUENTSPEECH_ACTIONS = ['change language', 'activate', 'deactivate', 'increase', 'decrease', 'bring']
    FLUENTSPEECH_OBJECTS = ['none',  'music', 'lights', 'volume',  'heat',   'lamp',    'newspaper',
                            'juice', 'socks', 'shoes',  'Chinese', 'Korean', 'English', 'German']
    FLUENTSPEECH_LOCATIONS = ['none', 'kitchen', 'bedroom', 'washroom']
    def __init__(self, root, label_type, train=True):
        super().__init__()
        self.root = os.path.join(root, 'speech')
        self.label_type = label_type
        self.train = train
        assert self.label_type in ['action', 'object', 'location']

        if train:
            train_path = os.path.join(self.root, 'fluent_speech_commands_dataset', 'data', 'train_data.csv')
            val_path = os.path.join(self.root, 'fluent_speech_commands_dataset', 'data', 'valid_data.csv')
            train_data = pd.read_csv(train_path)
            train_paths = list(train_data['path'])
            train_labels = list(train_data[self.label_type])
            val_data = pd.read_csv(val_path)
            val_paths = list(val_data['path'])
            val_labels = list(val_data[self.label_type])
            wav_paths = train_paths + val_paths
            labels = train_labels + val_labels
        else:
            test_path = os.path.join(self.root, 'fluent_speech_commands_dataset', 'data', 'test_data.csv')
            test_data = pd.read_csv(test_path)
            wav_paths = list(test_data['path'])
            labels = list(test_data[self.label_type])

        self.transform = T.Normalize([-31.809], [13.127])

        self.wav_paths = wav_paths
        self.labels = labels

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        wav_name = self.wav_paths[index]
        wav_path = os.path.join(self.root, 'fluent_speech_commands_dataset', wav_name)
        label = self.labels[index]
        if self.label_type == 'action':
            label = self.FLUENTSPEECH_ACTIONS.index(label)
        elif self.label_type == 'object':
            label = self.FLUENTSPEECH_OBJECTS.index(label)
        elif self.label_type == 'location':
            label = self.FLUENTSPEECH_LOCATIONS.index(label)

        wavform, sample_rate = torchaudio.load(wav_path)
        wavform = wavform[0].numpy()

        if len(wavform) > 150526:
            flip = (bool(random.getrandbits(1)) if self.train else True)
            padded = (wavform[:150526] if flip else wavform[-150526:])
        else:
            padded = np.zeros(150526)
            padded[:len(wavform)] = wavform  # pad w/ silence

        spectrum = librosa.feature.melspectrogram(
            y=padded,
            sr=sample_rate,
            hop_length=672,
            n_mels=224,
        )

        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)
        spectrum = self.transform(spectrum)

        return spectrum, int(label)


class GoogleSpeechCommand(Dataset):
    LABELS = ['eight', 'right', 'happy', 'three',  'yes',  'up',    'no',     'stop',   'on',       'four',    'nine',  'zero',
              'down',  'go',    'six',   'two',    'left', 'five',  'off',    'seven',  'one',      'cat',     'bird',  'marvin',
              'wow',   'tree',  'dog',   'sheila', 'bed',  'house', 'follow', 'visual', 'backward', 'forward', 'learn', '_background_noise_']
    def __init__(self, root, train=True):
        super().__init__()
        self.train = train
        self.root = os.path.join(root, 'speech', 'google_speech')

        if train:
            train_paths = []
            for path, _, files in os.walk(self.root):
                for name in files:
                    if name.endswith('wav'):
                        train_paths.append(os.path.join(path.split('/')[-1], name))
            val_paths = open(os.path.join(self.root, 'validation_list.txt'), 'r').readlines()
            test_paths = open(os.path.join(self.root, 'testing_list.txt'), 'r').readlines()
            train_paths = (set(train_paths) - set(val_paths) - set(test_paths))
            wav_paths = list(train_paths) + val_paths
        else:
            wav_paths = open(os.path.join(self.root, 'testing_list.txt'), 'r').readlines()

        self.transform = T.Normalize([-46.847], [19.151])

        self.wav_paths = [path.strip() for path in wav_paths]

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        wav_name = self.wav_paths[index]
        label_name = wav_name.split('/')[0].lower()
        label = self.LABELS.index(label_name)
        wav_path = os.path.join(self.root, wav_name)

        wavform, sample_rate = torchaudio.load(wav_path)
        wavform = wavform[0].numpy()

        if len(wavform) > 150526:
            flip = (bool(random.getrandbits(1)) if self.train else True)
            padded = (wavform[:150526] if flip else wavform[-150526:])
        else:
            padded = np.zeros(150526)
            padded[:len(wavform)] = wavform

        spectrum = librosa.feature.melspectrogram(
            y=padded,
            sr=sample_rate,
            hop_length=672,
            n_mels=224,
        )

        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)
        spectrum = self.transform(spectrum)

        return spectrum, int(label)


class VoxCeleb1(Dataset):
    MAX_LENGTH = 150526

    def __init__(self, root, train=True):
        super().__init__()
        self.root = os.path.join(root, 'speech', 'voxceleb1')
        self.wav_paths, speaker_strs = self.get_split(train)
        unique_speakers = sorted(set(speaker_strs))
        speaker_id_map = dict(zip(unique_speakers, range(len(unique_speakers))))
        self.speaker_ids = [speaker_id_map[sp] for sp in speaker_strs]
        self.train = train

        self.transform = T.Normalize([-37.075], [19.776])

    def __len__(self):
        return len(self.wav_paths)

    def get_split(self, train=True):
        split_file = os.path.join(self.root, 'iden_split.txt')
        with open(split_file, 'r') as fp:
            splits = fp.readlines()

        paths = defaultdict(lambda: [])
        for split in splits:
            spl, path = split.strip().split(' ')
            paths[spl].append(path)

        train_paths = paths['1'] + paths['2']
        test_paths = paths['3']
        train_speaker_ids = [p.split('/')[0] for p in train_paths]
        test_speaker_ids = [p.split('/')[0] for p in test_paths]
        if train:
            return train_paths, train_speaker_ids
        else:
            return test_paths, test_speaker_ids

    def __getitem__(self, index):
        wav_path = os.path.join(self.root, 'wav', self.wav_paths[index])
        speaker_id = self.speaker_ids[index]
        wavform, sample_rate = torchaudio.load(wav_path)
        wavform = wavform[0].numpy()
        if len(wavform) > self.MAX_LENGTH:
            flip = bool(random.getrandbits(1)) if self.train else True
            padded = (wavform[:self.MAX_LENGTH] if flip else wavform[-self.MAX_LENGTH:])
        else:
            padded = np.zeros(self.MAX_LENGTH)
            padded[:len(wavform)] = wavform  # pad w/ silence

        spectrum = librosa.feature.melspectrogram(
            y=padded,
            sr=sample_rate,
            hop_length=672,
            n_mels=224,
        )

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)
        spectrum = self.transform(spectrum)

        return spectrum, speaker_id


def get_dataset_config(dataset):
    if dataset == 'libri-speech':
        num_classes = 40
    elif dataset == 'audio-mnist':
        num_classes = 10
    elif dataset == 'fluent-speech-loc':
        num_classes = 4
    elif dataset == 'fluent-speech-obj':
        num_classes = 14
    elif dataset == 'fluent-speech-act':
        num_classes = 6
    elif dataset == 'google-speech':
        num_classes = 36
    elif dataset == 'voxceleb1':
        num_classes = 1251
    input_shape = (1, 224, 224)
    patch_size  = (16, 16)
    batch_size  = 64
    return dict(num_classes=num_classes,
                input_shape=input_shape,
                patch_size=patch_size,
                batch_size=batch_size)


def get_pretrain_dataset(dataset, datadir):
    if dataset == 'libri-speech':
        train = LibriSpeech(datadir, pre_train=True, download=False)
    else:
        raise Exception(f'No pretrain dataset: {dataset}')
    return train


def get_transfer_dataset(dataset, datadir):
    if dataset == 'libri-speech':
        val  = LibriSpeech(datadir, pre_train=False, train=True,  download=True)
        test = LibriSpeech(datadir, pre_train=False, train=False, download=True)
    elif dataset == 'audio-mnist':
        val  = AudioMNIST(datadir, train=True)
        test = AudioMNIST(datadir, train=False)
    elif dataset == 'fluent-speech-loc':
        val  = FluentSpeechCommand(datadir, 'location', train=True)
        test = FluentSpeechCommand(datadir, 'location', train=False)
    elif dataset == 'fluent-speech-obj':
        val  = FluentSpeechCommand(datadir, 'object', train=True)
        test = FluentSpeechCommand(datadir, 'object', train=False)
    elif dataset == 'fluent-speech-act':
        val  = FluentSpeechCommand(datadir, 'action', train=True)
        test = FluentSpeechCommand(datadir, 'action', train=False)
    elif dataset == 'google-speech':
        val  = GoogleSpeechCommand(datadir, train=True)
        test = GoogleSpeechCommand(datadir, train=False)
    elif dataset == 'voxceleb1':
        val  = VoxCeleb1(datadir, train=True)
        test = VoxCeleb1(datadir, train=False)
    else:
        raise Exception(f'No transfer dataset: {dataset}')
    return val, test

