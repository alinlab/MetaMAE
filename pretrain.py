import os
import random
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ignite.distributed as idist
from ignite.engine import Engine, Events, State
from ignite.utils import convert_tensor

import utils
import models
import data


def main(local_rank, args):
    seed = args.seed + local_rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    device = idist.device()
    logger, tb_logger = utils.get_logger(args)
    dataset = data.get_dataset(args.dataset, args.datadir, mode='pretrain')
    loader  = data.get_loader(args, dataset, mode='pretrain')

    args.num_epochs = args.num_iters // len(loader['train']) + 1

    model = models.get_model(args,
                             input_shape=dataset['input_shape'],
                             patch_size=dataset['patch_size'])
    model = idist.auto_model(model, sync_bn=True)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    optimizer = idist.auto_optim(optimizer)

    def training_step(engine, batch):
        model.train()
        batch = convert_tensor(batch, device=device, non_blocking=True)
        outputs = model(batch)
        optimizer.zero_grad()
        outputs['loss'].backward()
        optimizer.step()
        return outputs

    trainer = Engine(training_step)

    if logger is not None:
        trainer.logger = logger
        trainer.tb_logger = tb_logger
    trainer.add_event_handler(Events.ITERATION_COMPLETED, utils.log)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=args.save_freq), utils.save_checkpoint, args,
                              model=model, optimizer=optimizer)

    @trainer.on(Events.ITERATION_COMPLETED(once=args.num_iters+1000)) ##For stable termination
    def terminate(engine):
        print(f"-> terminate at iteration: {engine.state.iteration}")
        engine.terminate()

    trainer.run(loader['train'], max_epochs=args.num_epochs)
    if tb_logger is not None:
        tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='mscoco')
    parser.add_argument('--datadir', type=str, default='/data')

    parser.add_argument('--num-iters', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=6)

    parser.add_argument('--model', type=str, default='metamae')
    parser.add_argument('--backbone', type=str, default='dabs')
    parser.add_argument('--mask-ratio', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed-dim-dec', type=int, default=128)
    parser.add_argument('--num-layer-dec', type=int, default=4)

    parser.add_argument('--inner-lr', type=float, default=0.5)
    parser.add_argument('--reg-weight', type=float, default=0.1)
    parser.add_argument('--s-ratio', type=float, default=0.1)

    parser.add_argument('--use-first-order', action='store_true')

    parser.add_argument('--save-freq', type=int, default=10000)

    parser.add_argument('--master-port', type=int, default=2223)

    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    utils.setup_config(args)

    n = torch.cuda.device_count()
    if n == 1:
        with idist.Parallel() as parallel:
            parallel.run(main, args)
    else:
        with idist.Parallel(backend='nccl', nproc_per_node=n, master_port=os.environ.get('MASTER_PORT', args.master_port)) as parallel:
            parallel.run(main, args)

