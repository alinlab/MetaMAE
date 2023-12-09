import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchmetrics

import ignite.distributed as idist
from ignite.utils import convert_tensor, setup_logger

import utils
import models
import data


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def collect_features(model, loader, mode='feature', eval_type='global_pool'):
    model.eval()
    device = idist.device()
    X, Y = [], []
    for cnt, batch in enumerate(loader):
        x, y = convert_tensor(batch, device=device)
        if mode == 'feature':
            with torch.no_grad():
                x = model(x, mode='feature', eval=eval_type)
        else:
            x = model(x, mode='adaptation', eval=eval_type)
        X.append(x.detach().cpu())
        Y.append(y.detach().cpu())
        print(f'collect done: {cnt+1} / {len(loader)}', end='\r')
    X = torch.cat(X).detach().numpy()
    Y = torch.cat(Y).detach().numpy()
    return X, Y


@torch.no_grad()
def accuracy(output, target, topk=1):
    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
    return correct_k


@torch.no_grad()
def binary_accuracy(output, target):
    predicted = torch.round(output)
    correct = (predicted == target).float()
    accuracy = correct.sum()
    return accuracy


def evaluate(model, trainloader, testloader, num_classes):
    device = idist.device()
    logger = setup_logger(name='logging', filepath=os.path.join(args.ckptdir, f'lineval_{args.dataset}_{args.iter}_{args.transfer}.txt')) ##transfer, dataset should be considered
    logger.info(args)
    logger.info(' '.join(os.sys.argv))

    if args.dataset in data.EVAL_TOKENIZED:
        eval_type = 'tokenize'
    else:
        eval_type = 'global_pool'

    if args.transfer:
        X_train, Y_train = collect_features(model, trainloader, eval_type=eval_type, mode='adaptation')
        X_test,  Y_test  = collect_features(model, testloader,  eval_type=eval_type, mode='adaptation')
    else:
        X_train, Y_train = collect_features(model, trainloader, eval_type=eval_type)
        X_test,  Y_test  = collect_features(model, testloader,  eval_type=eval_type)

    train_dataset = FeatureDataset(X_train, Y_train)
    test_dataset  = FeatureDataset(X_test,  Y_test)
    trainloader = idist.auto_dataloader(train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        shuffle=True, drop_last=True,
                                        pin_memory=True)
    testloader  = idist.auto_dataloader(test_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        pin_memory=True)

    if num_classes > 2:
        classifier = nn.Linear(X_train.shape[-1], num_classes)
        if args.dataset in data.EVAL_AUROC:
            classifier = nn.Sequential(
                classifier,
                nn.Sigmoid()
            )
    elif num_classes == 0:
        classifier = nn.Linear(X_train.shape[-1], 1)
    else:
        classifier = nn.Sequential(
            nn.Linear(X_train.shape[-1], 1),
            nn.Sigmoid()
        )
    classifier = classifier.to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = idist.auto_optim(optimizer)

    if args.dataset in data.EVAL_SPEARMAN:
        criterion = torch.nn.MSELoss().to(device)
    elif num_classes > 2 and args.dataset not in data.EVAL_AUROC:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.BCELoss().to(device)

    if args.dataset in data.EVAL_F1:
        metric_fn = torchmetrics.F1Score(task='multiclass', num_classes=9, average='weighted') #waper-map only
    elif args.dataset in data.EVAL_AUROC:
        metric_fn = torchmetrics.AUROC(task='multilabel', num_labels=num_classes)

    for epoch in range(100):
        classifier.train()
        for batch in trainloader:
            x, y = convert_tensor(batch, device=device, non_blocking=True)
            logits = classifier(x)
            if eval_type == 'tokenize':
                logits = logits.reshape(-1, logits.shape[-1])
                y      = y.reshape(-1)
            if num_classes > 2 and args.dataset not in data.EVAL_AUROC:
                loss = criterion(logits, y)
            elif args.dataset in data.EVAL_SPEARMAN:
                loss = criterion(logits.reshape(-1), y.reshape(-1))
            else:
                loss = criterion(logits.reshape(-1), y.reshape(-1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        classifier.eval()

        top1_accuracy = 0
        cnt = 0
        logits_list = []
        y_list      = []
        with torch.no_grad():
            for batch in testloader:
                x, y = convert_tensor(batch, device=device)
                logits = classifier(x)
                if eval_type == 'tokenize':
                    logits = logits.reshape(-1, logits.shape[-1])
                    y      = y.reshape(-1)
                if args.dataset in data.EVAL_F1 + data.EVAL_SPEARMAN + data.EVAL_PEARSON + data.EVAL_AUROC:
                    logits_list += logits.detach().cpu().numpy().tolist()
                    y_list += y.detach().cpu().numpy().tolist()
                elif num_classes == 2:
                    top1_accuracy += binary_accuracy(logits.reshape(-1), y.reshape(-1)).item()
                    cnt += logits.shape[0]
                else:
                    top1_accuracy += accuracy(logits, y, topk=(1)).item()
                    cnt += logits.shape[0]
            if args.dataset in data.EVAL_F1:
                top1_accuracy = metric_fn(torch.tensor(logits_list), torch.tensor(y_list))
            elif args.dataset in data.EVAL_SPEARMAN:
                top1_accuracy = torchmetrics.functional.spearman_corrcoef(torch.tensor(logits_list).reshape(-1), torch.tensor(y_list).reshape(-1))
            elif args.dataset in data.EVAL_PEARSON:
                top1_accuracy = torchmetrics.functional.pearson_corrcoef(torch.tensor(logits_list).reshape(-1), torch.tensor(y_list).reshape(-1))
            elif args.dataset in data.EVAL_AUROC:
                top1_accuracy = metric_fn(torch.tensor(logits_list), torch.tensor(y_list))
            else:
                top1_accuracy /= cnt
            logger.info(f'[epoch {epoch}] [test acc top1 {top1_accuracy:.4f}]')


def main(local_rank, args):
    dataset = data.get_dataset(args.dataset, args.datadir, mode='transfer')
    loader  = data.get_loader(args, dataset, mode='transfer')
    args.batch_size = data.get_dataset(args.dataset, args.datadir, mode='transfer')['batch_size']

    model = models.get_model(args,
                             input_shape=dataset['input_shape'],
                             patch_size=dataset['patch_size'])
    model = idist.auto_model(model, sync_bn=True)

    ckpt = torch.load(os.path.join(args.ckptdir, f'ckpt-{args.iter}.pth'), map_location='cpu')
    model_state = ckpt['model']

    model_state_ = {}
    for k, v in model_state.items():
        if 'module' in k:
            model_state_[k[len('module.'):]] = v
        else:
            model_state_[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(model_state_, strict=False)
    print(f'missing: {missing_keys} | unexpected: {unexpected_keys}')

    evaluate(model, loader['val'], loader['test'], num_classes=dataset['num_classes'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckptdir', type=str, required=True)
    parser.add_argument('--iter', type=str, default='100k')
    parser.add_argument('--dataset', type=str, default='mismatched-caption')
    parser.add_argument('--datadir', type=str, default='/data')

    parser.add_argument('--num-workers', type=int, default=16)

    parser.add_argument('--model', type=str, default='mae')
    parser.add_argument('--backbone', type=str, default='dabs')
    parser.add_argument('--mask-ratio', type=float, default=0.75)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed-dim-dec', type=int, default=128)
    parser.add_argument('--num-layer-dec', type=int, default=4)

    parser.add_argument('--inner-lr', type=float, default=0.1)
    parser.add_argument('--reg-weight', type=float, default=1)
    parser.add_argument('--s-ratio', type=float, default=0.1)

    parser.add_argument('--use-first-order', action='store_true')

    parser.add_argument('--transfer', action='store_true')

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    utils.setup_config(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    with idist.Parallel() as parallel:
        parallel.run(main, args)

