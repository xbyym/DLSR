import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)  # Set model to training mode
    metric_logger = misc.MetricLogger(delimiter="  ")  # Logs and calculates training metrics, such as loss and learning rate
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20  # Frequency for printing log information

    accum_iter = args.accum_iter  # Gradient accumulation steps

    optimizer.zero_grad()  # Zero the parameter gradients

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, class_label) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Update learning rate using a per-iteration (instead of per-epoch) schedule if cosine learning rate is enabled
        if data_iter_step % accum_iter == 0 and args.cosine_lr:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Move data to the specified device (e.g., GPU)
        samples = samples.to(device, non_blocking=True)
        class_label = class_label.to(device, non_blocking=True)

        # Prepare the batch for conditional training if enabled
        if args.class_cond:
            batch = {"image": samples.permute([0, 2, 3, 1]), "class_label": class_label}
        else:
            batch = {"image": samples.permute([0, 2, 3, 1]), "class_label": torch.zeros_like(class_label)}
        
        # Forward pass through the model to compute loss
        loss, loss_dict = model(x=None, c=None, batch=batch)
        loss_value = loss.item()

        # Stop training if the loss is not finite (NaN or Inf)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Normalize loss by accumulation steps
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()  # Zero the gradients after updating

        torch.cuda.synchronize()  # Ensure all GPU operations are synchronized

        metric_logger.update(loss=loss_value)  # Update the metric logger with the loss value

        # Update learning rate in the logger
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # Log training loss and learning rate to tensorboard
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # Gather statistics from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def test_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()  # Set model to evaluation mode
    metric_logger = misc.MetricLogger(delimiter="  ")  # Logs and calculates evaluation metrics
    # Optionally add more metrics, e.g., accuracy

    with torch.no_grad():  # Disables gradient calculation for evaluation
        for data_iter_step, (samples, class_label) in enumerate(metric_logger.log_every(data_loader, print_freq=20, header='Test:')):
            # Move data to the specified device (e.g., GPU)
            samples = samples.to(device, non_blocking=True)
            class_label = class_label.to(device, non_blocking=True)

            # Prepare the batch for conditional evaluation if enabled
            if args.class_cond:
                batch = {"image": samples.permute([0, 2, 3, 1]), "class_label": class_label}
            else:
                batch = {"image": samples.permute([0, 2, 3, 1]), "class_label": torch.zeros_like(class_label)}
            
            # Forward pass to compute the output of the model
            outputs = model(samples)
            # Compute the loss for evaluation
            loss, loss_dict = model(x=None, c=None, batch=batch)

            metric_logger.update(loss=loss.item())  # Update the metric logger with the loss value
            # Optionally update other metrics here

    # Print and return the averaged metrics after going through all the data
    print("Averaged stats:", metric_logger) 
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
