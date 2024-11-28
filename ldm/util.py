import importlib

import numpy as np
from collections import abc
import torch.nn as nn
import multiprocessing as mp
from threading import Thread
from queue import Queue
import torch.nn.functional as F 
from inspect import isfunction
import torch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model



def calculate_similarity(sampled_Multi_Feature, z, indices, method='MFsim'):
    """
    Calculate similarity between sampled features and reference features.

    Args:
        sampled_Multi_Feature (Tensor): The sampled features.
        z (Tensor): The reference features.
        indices (list): Indices to use for block-wise similarity calculation.
        method (str): The method to use for similarity calculation ('mfsim' or 'mse').

    Returns:
        Tensor: The calculated similarity.
    """
    if method == 'MFsim':
        return cosine_similarity_per_block(sampled_Multi_Feature, z, indices)
    elif method == 'MSE':
        return mse_similarity(sampled_Multi_Feature, z)
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def cosine_similarity_per_block(sampled_rep, origan, indices):
    """
    Calculate cosine similarity per block.

    Args:
        sampled_rep (Tensor): Sampled representations.
        origan (Tensor): Original representations.
        indices (list): Indices defining the blocks.

    Returns:
        Tensor: MFsim value.
    """
    cosine_sims = []
    for start, end in zip(indices[:-1], indices[1:]):
        cos_sim = nn.functional.cosine_similarity(sampled_rep[:, start:end], origan[:, start:end], dim=1)
        cosine_sims.append(cos_sim)
    total_cosine_sim = torch.sum(torch.stack(cosine_sims), dim=0)
    return -total_cosine_sim / (len(indices) - 1)

def mse_similarity(sampled_rep, origan):
    """
    Calculate Mean Squared Error (MSE) for each sample.

    Args:
        sampled_rep (Tensor): Sampled representations.
        origan (Tensor): Original representations.

    Returns:
        Tensor: MSE similarity value for each sample.
    """
    # Calculate MSE for each sample without reduction across the batch
    mseloss = nn.functional.mse_loss(origan, sampled_rep, reduction='none')
    # Sum over the feature dimension to get the total error for each sample
    mseloss = mseloss.sum(dim=1)  

    return mseloss

class CustomGroupNorm(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices  # Indices to split the features into groups

    def forward(self, x):
        # x is assumed to be of shape [batch_size, num_features]
        group_normed_features = []
        start = 0  
        for end in self.indices:
            # Select the group
            group = x[:, start:end]
            
            # Compute mean and std for the group
            mean = group.mean(dim=1, keepdim=True)
            std = group.std(dim=1, keepdim=True) + 1e-5  # Adding a small constant to prevent division by zero
            # Normalize the group
            normalized_group = (group - mean) / std
            # Append the normalized group to the list
            group_normed_features.append(normalized_group)
            
            # Update start index for next group
            start = end
        
        # Concatenate all normalized groups along the feature dimension
        z = torch.cat(group_normed_features, dim=1)
        
        return z

def load_model(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "state_dict" not in pl_sd:
            pl_sd["state_dict"] = pl_sd["model"]
    else:
        pl_sd = {"state_dict": None}
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model

def load_model1(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        try:
            pl_sd = torch.load(ckpt, map_location="cpu")
            print("Model checkpoint loaded successfully.")
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")

        if "state_dict" not in pl_sd:
            pl_sd["state_dict"] = pl_sd["model"]
    else:
        pl_sd = {"state_dict": None}
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res
