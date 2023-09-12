import argparse
from pathlib import Path
import torch
import torch.nn as nn

import utils
import json
from datasets import build_dataset
from engine import train_one_epoch, evaluate
import TANet
import random
import numpy as np
from torch.cuda.amp import GradScaler


def get_args_parser():
    parser = argparse.ArgumentParser('Random Sampling SNN', add_help=False)
    parser.add_argument('--dataset', default='CIFAR-10', type=str, help='[CIFAR-10, CIFAR-100, IMAGENET]')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Spiking neuron parameters
    parser.add_argument('--T', default=4, type=int, help='time-steps')

    # Learning rate schedule parameters
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')

    # Network and graph parameters
    parser.add_argument('--graph-model', type=str, default='WS')
    parser.add_argument('--channels', type=int, default=96)
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--P', type=float, default=0.75)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--skip-ratio', type=float, default=0.25)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-skip', action='store_true')
    parser.add_argument('--every-node-dropout', action='store_true')
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--neuron-type', type=str, default='LIF')
    parser.add_argument('--tau', type=float, default=3.0)

    # Dataset parameters
    parser.add_argument('--data-path', default='../data', type=str, help='data path')
    parser.add_argument('--save-path', default='./output', type=str, help='path to save')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--graph-seed', default=15, type=int)
    parser.add_argument('--random-seed', default=15, type=int)
    parser.add_argument('--input-size', default=32, type=int)
    parser.add_argument('--evaluate-energy', action='store_true')
    parser.add_argument('--evaluate-spike', action='store_true')
    parser.add_argument('--checkpoint-path', default='./output', type=str)

    # distributed training parameters
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device('cuda')

    train_set, val_set, num_classes = build_dataset(args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    if utils.is_main_process():
        print(f'Preparing data!')

    train_sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=num_tasks, rank=global_rank,
                                                        shuffle=True)
    val_sampler = torch.utils.data.DistributedSampler(val_set, shuffle=False)

    train_data_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_set,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    if utils.is_main_process():
        print(f'Creating model')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.random_seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset == 'CIFAR-10' or args.dataset == 'CIFAR-100':
        model = TANet.Tiny(args, num_classes=num_classes)
    elif args.dataset == 'IMAGENET':
        model = TANet.Regular(args, num_classes=num_classes)
    else:
        raise NotImplementedError
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)

    output_dir = Path(args.save_path)

    if utils.is_main_process():
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        if args.save_path and utils.is_main_process():
            with (output_dir / 'log.txt').open("a") as f:
                f.write(f'number of params: {n_parameters}\n')

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 64.0
    args.lr = linear_scaled_lr
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    loss_scaler = GradScaler()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    criterion = torch.nn.CrossEntropyLoss()

    start_epoch = 0

    if utils.is_main_process():
        print(f'Start training for {args.epochs} epochs')
    max_accuracy = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, train_data_loader, optimizer, device, epoch, loss_scaler)

        scheduler.step()

        if args.save_path:
            checkpoint_path = output_dir / 'checkpoint.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)

        test_stats = evaluate(val_data_loader, model, device)
        print(f"Accuracy of the network on the {len(val_set)} test images: {test_stats['acc1']:.1f}%")
        if max_accuracy < test_stats['acc1']:
            max_accuracy = test_stats['acc1']
            if args.save_path:
                checkpoint_path = output_dir / 'best_checkpoint.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict() if args.distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        print(f'Max accuracy: {max_accuracy:.2f}')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch
                     }

        if args.save_path and utils.is_main_process():
            with (output_dir / 'log.txt').open("a") as f:
                f.write(json.dumps(log_stats) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Random Sampling SNN', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
    main(args)
