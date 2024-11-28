import copy
import importlib

import torch
import torch.nn as nn
from Pre_train_model.utils.misc_helper import to_device


class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)

    def build(self, mtype, kwargs):
        """
        Dynamically imports a module and initializes it with given parameters.

        Args:
            mtype (str): The fully qualified name of the module (including class).
            kwargs (dict): The keyword arguments for initializing the module.

        Returns:
            nn.Module: An instance of the specified module.
        """
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        """Moves the model to GPU and sets the device to CUDA."""
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        """Moves the model to CPU and sets the device to CPU."""
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        """
        Forward pass of the model.

        Args:
            input (dict): Input data dictionary containing tensors, usually the image tensor.

        Returns:
            dict: Updated input dictionary after passing through each module.
        """
        input = copy.copy(input)
        if input["image"].device != self.device:
            input = to_device(input, device=self.device)
        for name, submodule in self.named_children():  # Use named_children to get names and modules
            if name == 'reconstruction':  # Skip execution if the module name is "reconstruction"
                continue
            output = submodule(input)
            input.update(output)  # Update input dictionary for the next module

        return input

    def freeze_layer(self, module):
        """Freezes a layer by setting requires_grad to False for all parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode. This mainly affects modules such as Dropout or BatchNorm.

        Args:
            mode (bool): True for training mode, False for evaluation mode.

        Returns:
            nn.Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
