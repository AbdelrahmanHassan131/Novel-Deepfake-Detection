import os
import torch
import torch.nn as nn
from models.base.base_model import BaseModel
from models.shared.resnet import resnet50


class Wang2020_128Trainer(BaseModel):
    def name(self):
        return 'Wang2020_128'

    def __init__(self, opt):
        super(Wang2020_128Trainer, self).__init__(opt)

        # Pretrained backbone loading flag: skip local weights file or downloads if resuming or evaluating
        pretrained_flag = self.isTrain and not opt.continue_train and not getattr(opt, '_loading_checkpoint', False) and getattr(opt, 'pretrained', True)
        weights_path = getattr(opt, 'backbone_weights', None) if pretrained_flag else None
        self.model = resnet50(pretrained=pretrained_flag, weights_path=weights_path)

        # Head type and dropout configuration (linear historically defaults to 0.0 when rgb_dropout is absent)
        self.head_type = getattr(opt, 'rgb_head_type', getattr(opt, 'head_type', '128d'))
        default_dropout = 0.0 if self.head_type == 'linear' else 0.5
        self.dropout_rate = float(getattr(opt, 'rgb_dropout', default_dropout))
        embed_dim = getattr(opt, 'embed_dim', 128)

        if self.head_type == '128d':
            self.model.fc = nn.Sequential(
                nn.Linear(2048, embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(embed_dim, 1)
            )
        elif self.head_type == 'linear':
            self.model.fc = nn.Sequential(
                nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
                nn.Linear(2048, 1)
            )
        else:
            raise ValueError(f"Unknown rgb_head_type: {self.head_type!r}. Supported: ['128d', 'linear']")

        # Weight initialization for new training
        if self.isTrain and not opt.continue_train:
            init_gain = getattr(opt, 'init_gain', 0.02)
            if self.head_type == '128d':
                nn.init.normal_(self.model.fc[0].weight.data, 0.0, init_gain)
                if self.model.fc[0].bias is not None:
                    nn.init.constant_(self.model.fc[0].bias.data, 0.0)
                nn.init.normal_(self.model.fc[3].weight.data, 0.0, init_gain)
                if self.model.fc[3].bias is not None:
                    nn.init.constant_(self.model.fc[3].bias.data, 0.0)
            elif self.head_type == 'linear':
                lin_layer = self.model.fc[1] if isinstance(self.model.fc, nn.Sequential) and len(self.model.fc) > 1 else self.model.fc
                if hasattr(lin_layer, 'weight'):
                    nn.init.normal_(lin_layer.weight.data, 0.0, init_gain)
                    if lin_layer.bias is not None:
                        nn.init.constant_(lin_layer.bias.data, 0.0)

        # Fine-tuning parameter policy: full, head_only, layer4_and_head
        self.fine_tune_policy = getattr(opt, 'fine_tune_policy', 'full')
        self._apply_fine_tune_policy()

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self._init_optimizer(opt)

        if not self.isTrain and not getattr(opt, 'skip_load_networks', False):
            self.load_networks(getattr(opt, 'epoch', 'latest'))

        if getattr(opt, 'channels_last', False):
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.to(self.device)
        self.apply_bn_policy()

    def _apply_fine_tune_policy(self):
        """Set requires_grad on parameters according to fine_tune_policy."""
        policy = self.fine_tune_policy
        if policy == 'full':
            for p in self.model.parameters():
                p.requires_grad = True
        elif policy == 'head_only':
            for name, p in self.model.named_parameters():
                p.requires_grad = name.startswith('fc.')
        elif policy == 'layer4_and_head':
            for name, p in self.model.named_parameters():
                p.requires_grad = (name.startswith('fc.') or name.startswith('layer4.'))
        else:
            raise ValueError(f"Unknown fine_tune_policy: {policy!r}. Supported: ['full', 'head_only', 'layer4_and_head']")

    def _init_optimizer(self, opt):
        """Build optimizer with separate head/backbone groups, LR multipliers, and bias/norm decay policy."""
        weight_decay = getattr(opt, 'weight_decay', 0.0)
        momentum = getattr(opt, 'momentum', 0.0)
        backbone_lr_mult = float(getattr(opt, 'backbone_lr_mult', 1.0))
        optim_name = getattr(opt, 'optim', getattr(opt, 'optimizer', 'adam')).lower()
        decay_bias_norm = getattr(opt, 'decay_bias_norm', None)
        if decay_bias_norm is None:
            decay_bias_norm = (optim_name != 'adamw')

        head_decay, head_no_decay = [], []
        backbone_decay, backbone_no_decay = [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # 1D tensors (biases, normalization scale/bias) omit decay unless decay_bias_norm is True
            apply_decay = (p.ndim > 1 or decay_bias_norm)
            if name.startswith('fc.'):
                if apply_decay:
                    head_decay.append(p)
                else:
                    head_no_decay.append(p)
            else:
                if apply_decay:
                    backbone_decay.append(p)
                else:
                    backbone_no_decay.append(p)

        lr = float(getattr(opt, 'lr', 0.0001))
        beta1 = float(getattr(opt, 'beta1', 0.9))

        param_groups = []
        if head_decay:
            param_groups.append({'params': head_decay, 'lr': lr, 'weight_decay': weight_decay, 'name': 'head_decay'})
        if head_no_decay:
            param_groups.append({'params': head_no_decay, 'lr': lr, 'weight_decay': 0.0, 'name': 'head_no_decay'})

        backbone_lr = lr * backbone_lr_mult
        if backbone_decay:
            param_groups.append({'params': backbone_decay, 'lr': backbone_lr, 'weight_decay': weight_decay, 'name': 'backbone_decay'})
        if backbone_no_decay:
            param_groups.append({'params': backbone_no_decay, 'lr': backbone_lr, 'weight_decay': 0.0, 'name': 'backbone_no_decay'})

        if not param_groups:
            raise ValueError("[Wang2020_128Trainer] No trainable parameters found to optimize!")

        optim_name = getattr(opt, 'optim', getattr(opt, 'optimizer', 'adam')).lower()
        if optim_name == 'adam':
            self.optimizer = torch.optim.Adam(
                param_groups,
                lr=lr,
                betas=(beta1, 0.999),
                weight_decay=weight_decay,
            )
        elif optim_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=(beta1, 0.999),
                weight_decay=weight_decay,
            )
        elif optim_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                param_groups,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optim_name!r}. Supported: ['adam', 'adamw', 'sgd']")

    def _get_underlying_resnet(self):
        """Unwrap DDP or other wrappers to reach the underlying ResNet model."""
        m = self.model
        while hasattr(m, 'module'):
            m = m.module
        return m

    def apply_bn_policy(self):
        """Enforce bn_policy on BatchNorm layers."""
        if getattr(self.opt, 'bn_policy', 'train') == 'frozen':
            resnet = self._get_underlying_resnet()
            for m in resnet.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    m.eval()

    def train(self, mode=True):
        """Override train to maintain frozen BatchNorm statistics and frozen stages."""
        super().train(mode)
        if mode:
            self.apply_bn_policy()
            resnet = self._get_underlying_resnet()
            if self.fine_tune_policy == 'head_only':
                for name, module in resnet.named_children():
                    if name != 'fc':
                        module.eval()
            elif self.fine_tune_policy == 'layer4_and_head':
                for name, module in resnet.named_children():
                    if name not in ('fc', 'layer4'):
                        module.eval()
        return self

    def adjust_learning_rate(self, min_lr=1e-6):
        """Decay LR across all parameter groups while maintaining group ratios."""
        active = True
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                active = False
        return active

    def set_input(self, input):
        non_blocking = (self.device.type == 'cuda' and getattr(self.opt, 'pin_memory', True))
        inp = input[0].to(self.device, non_blocking=non_blocking)
        if getattr(self.opt, 'channels_last', False):
            inp = inp.to(memory_format=torch.channels_last)
        self.input = inp
        self.label = input[1].to(self.device, non_blocking=non_blocking).float()

    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
