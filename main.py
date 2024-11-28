import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine import train_one_epoch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from Pre_train_model.utils.misc_helper import update_config

def get_args_parser():
    parser = argparse.ArgumentParser('ldm training', add_help=False)
    # Training parameters
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=150, type=int, help='Number of training epochs')
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')

    # Configurations and dataset parameters
    parser.add_argument('--config', type=str, default='config/Config.yaml', help='Config file')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name (cifar10, cifar100, celeba)')
    parser.add_argument('--data_path', default='data/cifar10', type=str, help='Dataset path')
    parser.add_argument('--pre_ckpt_path', type=str, default='efficientnet-b4-6ed6700e.pth', help='Checkpoint path for the model')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.02, help='Weight decay')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--cosine_lr', action='store_true', help='Use cosine lr scheduling.')
    parser.add_argument('--warmup_epochs', default=0, type=int)

    # Logging and output parameters
    parser.add_argument('--output_dir', default='./output_dir', help='Directory to save outputs')
    parser.add_argument('--log_dir', default='./output_dir', help='Directory to save tensorboard logs')
    parser.add_argument('--device', default='cuda', help='Device to use for training/testing')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility')
    parser.add_argument('--resume', default='',help='resume from checkpoint')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=3, type=int, help='Rank of the process for distributed training')
    parser.add_argument('--dist_url', default='env://', help='URL to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true', help='Whether to initialize distributed training on IT partition')

    return parser

def main(args):
    # Initialize distributed mode
    misc.init_distributed_mode(args)

    print('Job dir:', os.path.dirname(os.path.realpath(__file__)))
    print("Args:", args)

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


    # Prepare logging
    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Define data transformations
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    if args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
    elif args.dataset == 'celeba':
        dataset_train = datasets.ImageFolder(root=args.data_path, transform=transform_train)
    else:
        raise ValueError("Unsupported dataset")

    # Distributed sampler for training dataset
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    # Data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=4,
        pin_memory=True, drop_last=True,
    )

    # Load model configuration and instantiate model
    config = OmegaConf.load(args.config)
    config = update_config(config)
    config.model.params.ckpt_path = args.pre_ckpt_path
    model = instantiate_from_config(config.model)
    model.to(device)

    # set arguments generation params
    args.class_cond = config.model.params.get("class_cond", False)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    # Load model if resuming
    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.epochs):
        if args.world_size > 1:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 25 == 0 or epoch + 1 == args.epochs):
            misc.save_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)