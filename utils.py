import os
import math

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ignite.utils import setup_logger, convert_tensor
import ignite.distributed as idist


def setup_config(args):
    if args.backbone == 'dabs':
        args.embed_dim_enc = 256
        args.num_head_enc = 8
        args.num_head_dec = 4
        args.num_layer_enc = 12
    else:
        raise NotImplementedError


def get_logger(args):
    if idist.get_rank() == 0:
        os.makedirs(args.logdir)
        logger = setup_logger(name='logging', filepath=os.path.join(args.logdir, 'log.txt'))
        logger.info(args)
        logger.info(' '.join(os.sys.argv))
        tb_logger = SummaryWriter(log_dir=args.logdir)
    else:
        logger, tb_logger = None, None

    idist.barrier()
    return logger, tb_logger


@idist.one_rank_only()
def log(engine):
    if engine.state.iteration % 10 == 0:
        engine.logger.info(f'[Epoch {engine.state.epoch:4d}] '
                           f'[Iter {engine.state.iteration:6d}] '
                           f'[Loss {engine.state.output["loss"].item():.4f}]')
        for k, v in engine.state.output.items():
            engine.tb_logger.add_scalar(k, v, engine.state.iteration)


@idist.one_rank_only()
def save_checkpoint(engine, args, **kwargs):
    state = { k: v.state_dict() for k, v in kwargs.items() }
    state['engine'] = engine.state_dict()
    torch.save(state, os.path.join(args.logdir, f'ckpt-{engine.state.iteration//1000}k.pth'))


def collect_features(model, loader, mode='feature'):
    model.eval()
    device = idist.device()
    X, Y = [], []
    for cnt, batch in enumerate(loader):
        x, y = convert_tensor(batch, device=device)
        if mode == 'feature':
            with torch.no_grad():
                x = model(x, mode=mode)
        else:
            x = model(x, mode=mode)
            model.zero_grad()
        X.append(x.detach().cpu())
        Y.append(y.detach().cpu())
        print(f'collect done: {cnt+1} / {len(loader)}', end='\r')
    X = torch.cat(X).detach()
    Y = torch.cat(Y).detach()
    return X, Y

