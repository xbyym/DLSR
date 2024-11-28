import argparse
import numpy as np
import os
import torch.nn.functional as F 
import torch 
import torch.backends.cudnn as cudnn 
from torch.utils.tensorboard import SummaryWriter 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 
import torchvision
import timm
from torch.utils.data import DataLoader 
assert timm.__version__ == "0.3.2"  # Version check to ensure compatibility with specific features of timm library
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from Gen_Model import Generative_Mul_Feature
from sklearn.metrics import roc_auc_score

import logging

def get_args_parser():
    parser = argparse.ArgumentParser('DLSR_Test', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=200, type=int, help='Total number of training epochs')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='Input image size')

    # Pre-trained encoder parameters
    parser.add_argument('--use_Mul_feature', default=True, action='store_true', help='Use Feature as condition.')
    parser.add_argument('--use_class_label', action='store_true', help='Use class label as condition.')

    # LDM parameters
    parser.add_argument('--pretrained_ldm_ckpt',  default='', type=str, help='Path to pre-trained LDM checkpoint')
    parser.add_argument('--pretrained_ldm_cfg',  default='config/Config.yaml', type=str, help='Path to pre-trained LDM configuration file')
    parser.add_argument('--ldm_steps', default=200, type=int, help='Number of steps for LDM generation')
    parser.add_argument('--eta', default=1.0, type=float, help='Eta value for LDM sampling')
    parser.add_argument('--log_dir', default='./output_dir_logging', help='Directory where to save logs')

    # Multi-scale features generation parameters
    parser.add_argument('--evaluate', action='store_true', help="Perform only evaluation")
    parser.add_argument('--eval_freq', type=int, default=40, help='Evaluation frequency')
    parser.add_argument('--temp', default=6.0, type=float, help='Sampling temperature')
    parser.add_argument('--similarity_type', type=str, choices=['MSE', 'MFsim'], default='MFsim',
                    help='Select similarity calculation method: MSE or MFsim')
    parser.add_argument('--num_iter', default=16, type=int, help='Number of iterations for generation')
    parser.add_argument('--num_images', default=50, type=int, help='Number of images to generate')
    parser.add_argument('--cfg', default=0.0, type=float, help='Configuration value for generation')
    parser.add_argument('--subset_timesteps_num', default=3, type=int, help='Use subset_timesteps instead of the original timesteps variable')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay for regularization (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='Learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='Number of epochs to warmup learning rate')

    # Dataset parameters
    parser.add_argument('--dataset', default='cifar10', type=str,
                        help='Dataset to use (e.g., cifar10, cifar100, celeba)')
    parser.add_argument('--data_path', default='', type=str,
                        help='Path to dataset')
    parser.add_argument('--augmentation', default='randresizedcrop', type=str,
                        help='Augmentation type')

    parser.add_argument('--output_dir', default='./output_dir_logging',
                        help='Path where to save outputs, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing (e.g., cuda or cpu)')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility')
    parser.add_argument('--resume', default='', help='Path to resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch number for resuming training')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=3, type=int, help='Local rank for distributed training')
    parser.add_argument('--dist_on_itp', action='store_true', help='Use distributed mode if set')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser

def evaluate_ood_detection(model, total_id, ood_data_loaders, device, dataset, config_path):
    print(f"Evaluating OOD detection for dataset: {dataset} with config: {config_path}")

    # Calculate the threshold for ID vs. OOD detection
    threshold_total_cos_sim = np.percentile(total_id, 95)

    for idx, data_loader_ood in enumerate(ood_data_loaders, start=1):
        total_ood = compute_metrics(model, data_loader_ood, device)

        # Calculate False Positive Rate at 95% for OOD detection
        ood_detected_total_cos_sim = np.sum(np.array(total_ood) < threshold_total_cos_sim) / len(total_ood)
        print(f"Evaluating OOD Dataset FPR95% {idx}:")
        print(f"Total Cosine Similarity FPR95: {ood_detected_total_cos_sim:.4f}")

        # Calculate AUROC for OOD detection
        auroc_total = roc_auc_score([0]*len(total_id) + [1]*len(total_ood), np.concatenate([total_id, total_ood]))
        print(f"Total Cosine Similarity AUROC: {auroc_total:.4f}")
        print("--")


def compute_metrics(model, data_loader, device):
    model.eval()
    total_cos_sim_list = []

    with torch.no_grad():
        for samples, class_label in data_loader:
            samples = samples.to(device, non_blocking=True)
            class_label = class_label.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                total_cosine = model(samples,class_label)
            total_cos_sim_list.extend(total_cosine.cpu().numpy())

    return total_cos_sim_list


def main(args):
    print("Program started")

    misc.init_distributed_mode(args)
    print('Job directory: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Initialize log writer
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # Transformations for the training and testing datasets
    transform_train = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the dataset based on the input argument
    if args.dataset == 'cifar10':
        dataset_train = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_train)
        dataset_test0 = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_train) 
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train) 
        dataset_test = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_train) 
        dataset_test0 = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_train) 
    elif args.dataset == 'celeba': 
        dataset_train = datasets.ImageFolder(root=args.data_path, transform=transform_train) 
        dataset_test = datasets.ImageFolder(root=args.data_path, transform=transform_train)
        dataset_test0 = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_train) 

    # Additional datasets for OOD detection
    dataset_test1 = datasets.ImageFolder(root='', transform=transform_train)
    dataset_test2 = torchvision.datasets.SVHN(root='', split='test',
                                              download=True, transform=transform_train) 
    dataset_test3 = datasets.ImageFolder(root='', transform=transform_train)
    dataset_test4 = datasets.ImageFolder(root='', transform=transform_train)
    dataset_test5 = datasets.ImageFolder(root='', transform=transform_train)
    dataset_test6 = datasets.ImageFolder(root='', transform=transform_train)

    print(dataset_train)

    # Setup data samplers for distributed training
    if True:  # args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_test = torch.utils.data.RandomSampler(dataset_test)

    # Create DataLoader for training and testing
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )
    data_loader_test = DataLoader(dataset_test,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=args.pin_mem,drop_last=True)
    data_loader_ood0 = DataLoader(dataset_test0, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem,drop_last=True)
    data_loader_ood1 = DataLoader(dataset_test1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem,drop_last=True)
    data_loader_ood2 = DataLoader(dataset_test2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem,drop_last=True)
    data_loader_ood3 = DataLoader(dataset_test3, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem,drop_last=True)
    data_loader_ood4 = DataLoader(dataset_test4, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem,drop_last=True)
    data_loader_ood5 = DataLoader(dataset_test5, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem,drop_last=True) 
    data_loader_ood6 = DataLoader(dataset_test6, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_mem,drop_last=True)

    # Model Initialization
    model = Generative_Mul_Feature(
        use_Mul_feature=args.use_Mul_feature,
        use_class_label=args.use_class_label,
        pretrained_ldm_ckpt=args.pretrained_ldm_ckpt,
        pretrained_ldm_cfg=args.pretrained_ldm_cfg,
        subset_timesteps_num=args.subset_timesteps_num,
        similarity_type=args.similarity_type  # New parameter
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size() 
    
    # Calculate effective learning rate if not provided
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # Set up distributed data parallel if applicable
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Log parameters
    n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print("Number of trainable parameters: {}M".format(n_params / 1e6))
    if global_rank == 0:
        log_writer.add_scalar('num_params', n_params / 1e6, 0)
    
    # following timm: set weight decay as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay) 
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95)) 
    print(optimizer) 
    loss_scaler = NativeScaler() 

    # Load model from checkpoint if applicable
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler) 

    # Put all OOD data loaders into a list for evaluation
    ood_data_loaders = [data_loader_ood0, data_loader_ood1, data_loader_ood2, data_loader_ood3, data_loader_ood4, data_loader_ood5, data_loader_ood6] 

    # Evaluate OOD detection with given model and data loaders
    total_id = compute_metrics(model, data_loader_test, device)
    evaluate_ood_detection(model, total_id, ood_data_loaders, device, args.dataset, args.pretrained_ldm_cfg)

if __name__ == '__main__': 
    args = get_args_parser() 
    args = args.parse_args() 

    main(args)
