import os
import json
import random
from copy import deepcopy

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms as T
import torchvision.datasets as VD

from transformers import AutoTokenizer


class MSCOCO(Dataset):
    def __init__(self, root, train=True):
        super().__init__()
        self.train = train
        self.root = os.path.join(root, 'COCO')

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=self.root)

        image_dir_name = ('train2017' if self.train else 'val2017')
        image_dir = os.path.join(self.root, image_dir_name)

        captions, labels, coco_cat_id_to_label = self.load_coco()
        self.paths, self.captions, self.labels = [], [], []
        for image_id in captions.keys():
            if image_id in labels:
                self.paths.append(os.path.join(image_dir, '%012d.jpg' % image_id))
                self.captions.append(captions[image_id])
                self.labels.append(labels[image_id])

        self.coco_cat_id_to_label = coco_cat_id_to_label  # not needed, but maps original classes to enumerated ones

        self.transform = T.Compose([T.Resize((640, 480)),
                                    T.CenterCrop((480, 480)),
                                    T.Resize((224, 224)),
                                    T.ToTensor()])

    def load_coco(self):
        annotation_name = ('instances_train2017.json' if self.train else 'instances_val2017.json')
        annotation_path = os.path.join(self.root, 'annotations', annotation_name)

        caption_name = ('captions_train2017.json' if self.train else 'captions_val2017.json')
        caption_path = os.path.join(self.root, 'annotations', caption_name)

        with open(annotation_path, 'r') as json_file:
            annotations = json.load(json_file)
            categories = annotations['categories']

        category_ids = [cat['id'] for cat in categories]
        coco_cat_id_to_label = dict(zip(category_ids, range(len(categories))))

        label_annotations = {}
        for annotation in annotations['annotations']:
            label_annotations[annotation['image_id']] = coco_cat_id_to_label[annotation['category_id']]

        with open(caption_path, 'r') as json_file:
            captions = json.load(json_file)['annotations']
            caption_annotations = {c['image_id']: c['caption'] for c in captions}

        return caption_annotations, label_annotations, coco_cat_id_to_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]

        caption = self.captions[index]
        caption = self.tokenizer.encode(
            caption,
            max_length=32,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        ).squeeze(0)

        image = Image.open(path).convert(mode='RGB')
        image = self.transform(image)
        return (image.float(), caption.long()), label


class MismatchedCaption(MSCOCO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_swap()

    def static_swap(self):
        '''Randomly swaps captions for each image with 50% probability. Revises the labels to reflect swapped or original.'''
        random.seed(42)

        # Accumulate indices to be swapped with 50% probability.
        orig_indices = []
        sames, swaps = 0, 0
        for index in range(len(self)):
            if random.random() < 0.5:
                orig_indices += [index]
                self.labels[index] = 0
                swaps += 1
            else:
                self.labels[index] = 1
                sames += 1

        # Roll indices.
        roll_length = random.randint(1, len(orig_indices))
        new_indices = orig_indices[roll_length:] + orig_indices[:roll_length]

        # Reassign captions.
        captions_copy = deepcopy(self.captions)  # deepcopy here to avoid overwriting originals
        for orig_index, new_index in zip(orig_indices, new_indices):
            self.captions[orig_index] = captions_copy[new_index]

        print(f'{sames} examples kept same, {swaps} examples swapped.')


class VQA(VD.vision.VisionDataset):
    def __init__(self, root, train=True):
        self.root = os.path.join(root, 'captioned_images', 'vqa')
        super().__init__(self.root)
        self.split = 'train' if train else 'val'
        self.data_root = os.path.join(self.root, 'coco-vqa')
        self.image_root = os.path.join(self.data_root, f'{self.split}2014')

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=self.root)

        self.id2image_file = {}
        self.id2question = {}
        self.id2choices = {}
        self.annotations = []
        self.transform = T.Compose([T.Resize((640, 480)),
                                    T.CenterCrop((480, 480)),
                                    T.Resize((224, 224)),
                                    T.ToTensor(),])
        self._build_index()
        self._static_swap()

    def _build_index(self):
        print('Building index...')
        anns = json.load(open(os.path.join(self.data_root, f'annotations/captions_{self.split}2014.json'), 'r'))
        vqa_anns = json.load(open(os.path.join(self.data_root, f'mscoco_{self.split}2014_annotations.json'), 'r'))
        questions = json.load(
            open(os.path.join(self.data_root, f'MultipleChoice_mscoco_{self.split}2014_questions.json'), 'r')
        )

        for image in anns['images']:
            self.id2image_file[image['id']] = image['file_name']

        for question in questions['questions']:
            self.id2question[question['question_id']] = question['question']
            self.id2choices[question['question_id']] = question['multiple_choices']

        self.annotations = vqa_anns['annotations']
        print('Finished index')

    def _static_swap(self):
        swaps, sames, trash = 0, 0, 0
        annotations = []
        for index, annotation in enumerate(self.annotations):
            question_id = annotation['question_id']
            answer = annotation['multiple_choice_answer']

            # Make sure the answer is in the list of choices.
            choices = self.id2choices[question_id]
            if answer not in choices:
                trash += 1
                continue

            # Swap in wrong answer with 50% probability.
            if random.random() < 0.5:
                choices = [choice for choice in choices if choice != answer]
                answer = random.choice(choices)

                # Overwrite (in place but we reassign the whole thing later anyways).
                self.annotations[index]['multiple_choice_answer'] = answer
                self.annotations[index]['label'] = 0
                swaps += 1
            else:
                self.annotations[index]['label'] = 1
                sames += 1

            annotations += [self.annotations[index]]

        self.annotations = annotations
        print(f'{swaps} answers swapped, {sames} answers kept same, {trash} answers thrown out.')

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]

        image = Image.open(os.path.join(self.image_root, self.id2image_file[annotation['image_id']])).convert('RGB')
        image = self.transform(image)

        question = self.id2question[annotation['question_id']]
        answer = annotation['multiple_choice_answer']

        label = annotation['label']
        tokens = self.tokenizer.encode(
            question,
            answer,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).squeeze(0)

        return (image, tokens), torch.tensor(label, dtype=torch.long)



def get_dataset_config(dataset):
    if dataset == 'mscoco':
        num_classes = 80
    elif dataset in ['mismatched-caption', 'vqa']:
        num_classes = 2
    input_shape = ((3, 224, 224), ((30552,), 32))
    patch_size  = ((16, 16), (1,))
    batch_size  = 64
    return dict(num_classes=num_classes,
                input_shape=input_shape,
                patch_size=patch_size,
                batch_size=batch_size)


def get_pretrain_dataset(dataset, datadir):
    if dataset == 'mscoco':
        train = MSCOCO(datadir, train=True)
    else:
        raise Exception(f'No pretrain dataset: {dataset}')
    return train


def get_transfer_dataset(dataset, datadir):
    if dataset == 'mismatched-caption':
        val  = MismatchedCaption(datadir, train=True)
        test = MismatchedCaption(datadir, train=False)
    elif dataset == 'vqa':
        val  = VQA(datadir, train=True)
        test = VQA(datadir, train=False)
    else:
        raise Exception(f'No transfer dataset: {dataset}')
    return val, test

