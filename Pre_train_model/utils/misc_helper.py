import importlib
import logging 
import os 
import random 
import shutil 
from collections.abc import Mapping 
from datetime import datetime 

import numpy as np 
import torch 
import torch.distributed as dist 


def basicConfig(*args, **kwargs): 
    return 

# To prevent duplicate logs, we mask this baseConfig setting 
logging.basicConfig = basicConfig 

def create_logger(name, log_file, level=logging.INFO): 
    """
    Create a logger that writes log messages to both a log file and the console.

    Args:
        name (str): The name of the logger.
        log_file (str): Path to the log file where log messages will be saved.
        level (int): Logging level (e.g., logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file) 
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level) 
    log.addHandler(fh)
    log.addHandler(sh)
    return log

def get_current_time():
    """
    Get the current time as a formatted string.

    Returns:
        str: Current time in "YYYYMMDD_HHMMSS" format.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time

class AverageMeter(object):
    """
    Computes and stores the average and current value of metrics.
    """

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        """
        Reset the meter to its initial state.
        """
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0 
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        """
        Update the meter with a new value.

        Args:
            val (float): The value to add.
            num (int): The number of occurrences (default is 1).
        """
        if self.length > 0:
            # Currently assert num==1 to avoid misuse, refine when there are explicit requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

def save_checkpoint(state, config):
    """
    Save the current model checkpoint.

    Args:
        state (dict): State dictionary containing model parameters and metadata.
        config (dict): Configuration dictionary with save path.
    """
    folder = config.save_path

    # Save the current model state
    checkpoint_path = os.path.join(folder, f"ckpt_{state['epoch']}.pth.tar")
    torch.save(state, checkpoint_path)

    # Save as "latest_checkpoint.pth.tar" if always_save is set
    if config.saver.get("always_save", True):  # By default, always save
        shutil.copyfile(checkpoint_path, os.path.join(folder, "latest_checkpoint.pth.tar"))

def load_state(path, model, optimizer=None):
    """
    Load a checkpoint into the model and optionally the optimizer.

    Args:
        path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load the state into.

    Returns:
        tuple: Best metric and epoch if optimizer is provided, otherwise None.
    """
    rank = dist.get_rank()

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # Fix size mismatch error by ignoring keys with mismatched shapes
        ignore_keys = []
        for k, v in checkpoint["state_dict"].items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    if rank == 0:
                        print(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            )
                        )

        for k in ignore_keys:
            checkpoint["state_dict"].pop(k)

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if rank == 0:
            ckpt_keys = set(checkpoint["state_dict"].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            best_metric = checkpoint["best_metric"]
            epoch = checkpoint["epoch"]
            if rank == 0:
                print(
                    "=> also loaded optimizer from checkpoint '{}' (Epoch {})".format(
                        path, epoch
                    )
                )
            return best_metric, epoch
    else:
        if rank == 0:
            print("=> no checkpoint found at '{}'".format(path))

def set_random_seed(seed=233, reproduce=False):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value.
        reproduce (bool): Whether to make the results fully reproducible.
    """
    np.random.seed(seed)
    torch.manual_seed(seed ** 2)
    torch.cuda.manual_seed(seed ** 3)
    random.seed(seed ** 4)

    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

def to_device(input, device="cuda", dtype=None):
    """
    Transfer data between devices.

    Args:
        input (dict): Dictionary of data to be transferred.
        device (str): Target device (default is "cuda").
        dtype (torch.dtype, optional): Data type for tensor.

    Returns:
        dict: Data dictionary with all tensors moved to the specified device.
    """
    if "image" in input:
        input["image"] = input["image"].to(dtype=dtype)

    def transfer(x):
        if torch.is_tensor(x):
            return x.to(device=device)
        elif isinstance(x, list):
            return [transfer(_) for _ in x]
        elif isinstance(x, Mapping):
            return type(x)({k: transfer(v) for k, v in x.items()})
        else:
            return x

    return {k: transfer(v) for k, v in input.items()}

def update_config(config):
    """
    Update model configuration based on pretrained encoder configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Updated configuration.
    """
    # Get model configuration
    model_config = config['model']['params']

    # Get pretrained encoder configuration from model configuration
    pretrained_enc_config = model_config['pretrained_enc_config']

    # Update planes & strides
    backbone_path, backbone_type = pretrained_enc_config[0].type.rsplit(".", 1)
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outblocks = None
    if "efficientnet" in backbone_type:
        outblocks = []
    outstrides = []
    outplanes = []
    for layer in pretrained_enc_config[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        if "efficientnet" in backbone_type:
            outblocks.append(backbone["blocks"][idx])
        outstrides.append(backbone["strides"][idx])
        outplanes.append(backbone["planes"][idx])
    if "efficientnet" in backbone_type:
        pretrained_enc_config[0].kwargs.pop("outlayers")
        pretrained_enc_config[0].kwargs.outblocks = outblocks
    pretrained_enc_config[0].kwargs.outstrides = outstrides
    pretrained_enc_config[1].kwargs.outplanes = [sum(outplanes)]

    return config
