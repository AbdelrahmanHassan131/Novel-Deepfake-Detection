# Refactored BaseModel for the model layer.
# Preserves all original behavior from MyModels/networks/base_model.py
# and adds centralized checkpoint support and DDP preparation.

import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


class BaseModel(nn.Module):
    """
    Base class for all deepfake detection models.

    Provides:
        - Shared device handling (DDP-compatible, no global state)
        - Centralized checkpoint save/load with full training state
        - Model metadata via name()
        - Common train/eval/test interface

    Subclasses must:
        - Set self.model to the nn.Module used for forward pass
        - Implement forward(), set_input(), optimize_parameters()
        - Optionally override name()
    """

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        print("used device is ", self.device)

    def name(self):
        return 'BaseModel'

    # ------------------------------------------------------------------
    # Original save/load (preserved for backward compatibility)
    # ------------------------------------------------------------------

    def save_networks(self, epoch):
        """Save model and optimizer state (original format)."""
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }

        torch.save(state_dict, save_path)

    # load models from the disk
    def load_networks(self, epoch):
        """Load model and optimizer state (original format)."""
        load_filename = 'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)

        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        self.model.load_state_dict(state_dict['model'])
        self.total_steps = state_dict['total_steps']

        if self.isTrain and not self.opt.new_optim:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            # move optimizer state to GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            for g in self.optimizer.param_groups:
                g['lr'] = self.opt.lr

    # ------------------------------------------------------------------
    # Extended checkpoint support (for resume training / DDP)
    # ------------------------------------------------------------------

    def save_checkpoint(self, filepath, epoch=None, best_metric=None,
                        scheduler=None):
        """
        Save a full training checkpoint for resume support.

        Saves:
            - model_state_dict
            - optimizer_state_dict
            - scheduler_state_dict (if provided)
            - epoch
            - best_metric
            - total_steps
            - model_name (metadata)

        Args:
            filepath: Full path for the checkpoint file.
            epoch: Current epoch number.
            best_metric: Best validation metric achieved so far.
            scheduler: Learning rate scheduler (optional).
        """
        # Handle DDP-wrapped models: unwrap to get raw state_dict
        model_to_save = self.model
        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module

        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
                if hasattr(self, 'optimizer') else None,
            'epoch': epoch,
            'best_metric': best_metric,
            'total_steps': self.total_steps,
            'model_name': self.name(),
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        else:
            checkpoint['scheduler_state_dict'] = None

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved to {filepath}')

    def load_checkpoint(self, filepath, scheduler=None, strict=True):
        """
        Load a full training checkpoint for resume support.

        Restores:
            - model weights
            - optimizer state
            - scheduler state (if provided and present in checkpoint)
            - epoch number
            - best_metric

        Args:
            filepath: Full path to the checkpoint file.
            scheduler: Learning rate scheduler to restore (optional).
            strict: Whether to strictly enforce state_dict key matching.

        Returns:
            dict with 'epoch', 'best_metric' from the checkpoint.
        """
        print(f'Loading checkpoint from {filepath}')
        checkpoint = torch.load(filepath, map_location=self.device)

        # Handle DDP-wrapped models
        model_to_load = self.model
        if hasattr(model_to_load, 'module'):
            model_to_load = model_to_load.module

        model_to_load.load_state_dict(
            checkpoint['model_state_dict'], strict=strict)

        if (hasattr(self, 'optimizer') and self.optimizer is not None
                and checkpoint.get('optimizer_state_dict') is not None):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move optimizer state to correct device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

        if (scheduler is not None
                and checkpoint.get('scheduler_state_dict') is not None):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.total_steps = checkpoint.get('total_steps', 0)

        return {
            'epoch': checkpoint.get('epoch', 0),
            'best_metric': checkpoint.get('best_metric', None),
        }

    # ------------------------------------------------------------------
    # Common interface
    # ------------------------------------------------------------------

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def get_model(self):
        """Return the underlying nn.Module (unwrapped from DDP if needed)."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model


def init_weights(net, init_type='normal', gain=0.02):
    """Initialize network weights.

    Preserved exactly from the original implementation.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
