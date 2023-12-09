import math

import ignite.distributed as idist

from . import rgbimage
from . import msimage
from . import timeseries
from . import speech
from . import tabular
from . import token
from . import text
from . import captionedimage
from . import utils


D_RGB_IMAGE =       ['imagenet32', 'cifar10',      'cub',      'vggflower',
                     'dtd',        'traffic-sign', 'aircraft', 'wafer-map']
D_MS_IMAGE =        ['eurosat']
D_TIME_SERIES =     ['pamap2']
D_SPEECH =          ['libri-speech',      'audio-mnist',   'fluent-speech-loc', 'fluent-speech-obj',
                     'fluent-speech-act', 'google-speech', 'voxceleb1']
D_TABULAR =         ['higgs']
D_TOKEN =           ['genomics', 'genomics-id',         'genomics-ood', 'pfam',
                     'scop',     'secondary-structure', 'stability',    'fluorescence']
D_CAPTIONED_IMAGE = ['mscoco', 'mismatched-caption', 'vqa']

EVAL_TOKENIZED = ['secondary-structure']
EVAL_SPEARMAN  = ['stability', 'fluorescence']
EVAL_F1        = ['wafer-map']


def _modality_call_fn(fn, dataset, *args, **kwargs):
    if dataset in D_RGB_IMAGE:
        return getattr(rgbimage, fn)(dataset, *args, **kwargs)
    elif dataset in D_MS_IMAGE:
        return getattr(msimage, fn)(dataset, *args, **kwargs)
    elif dataset in D_TIME_SERIES:
        return getattr(timeseries, fn)(dataset, *args, **kwargs)
    elif dataset in D_SPEECH:
        return getattr(speech, fn)(dataset, *args, **kwargs)
    elif dataset in D_TABULAR:
        return getattr(tabular, fn)(dataset, *args, **kwargs)
    elif dataset in D_TOKEN:
        return getattr(token, fn)(dataset, *args, **kwargs)
    elif dataset in D_CAPTIONED_IMAGE:
        return getattr(captionedimage, fn)(dataset, *args, **kwargs)
    else:
        raise NotImplementedError


def get_dataset(dataset, datadir, mode='pretrain'):
    data_dict = _modality_call_fn('get_dataset_config', dataset)
    data_dict['n_modality'] = 2 if dataset in D_CAPTIONED_IMAGE else 1
    if mode == 'pretrain':
        data_dict['train'] = _modality_call_fn('get_pretrain_dataset', dataset, datadir=datadir)
    else:
        val, test = _modality_call_fn('get_transfer_dataset', dataset, datadir=datadir)
        data_dict['val']  = val
        data_dict['test'] = test

    return data_dict


def get_loader(args, dataset, mode='pretrain'):
    loader = {}
    if mode == 'pretrain':
        loader['train'] = idist.auto_dataloader(dataset['train'],
                                                batch_size=dataset['batch_size'],
                                                num_workers=args.num_workers,
                                                shuffle=True, drop_last=True,
                                                pin_memory=True)
    else:
        for split in ['val', 'test']:
            loader[split] = idist.auto_dataloader(dataset[split],
                                                batch_size=dataset['batch_size'],
                                                num_workers=args.num_workers,
                                                pin_memory=True)

    return loader