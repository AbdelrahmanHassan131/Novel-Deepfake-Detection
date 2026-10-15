"""DRCT_128 Model Trainer.

Integrates DRCT128Model with the training engine, optimizer lifecycle,
loss calculation, and DDP/AMP support.
"""
from typing import Any, Optional
import torch
import torch.nn as nn
from models.base.base_model import BaseModel
from models.drct.drct_model import DRCT128Model
from models.drct.loss import DRCTCombinedLoss


class DRCT_128Trainer(BaseModel):
    """Trainer class for DRCT model with 128-dimensional embedding head."""

    def name(self) -> str:
        return 'DRCT_128'

    def __init__(self, opt: Any):
        super(DRCT_128Trainer, self).__init__(opt)

        embed_dim = int(getattr(opt, 'embed_dim', 128))
        if embed_dim != 128:
            raise ValueError(f"DRCT_128 requires embed_dim=128, got embed_dim={embed_dim}.")

        # Backbone and pretraining options
        backbone_name = getattr(opt, 'drct_backbone', getattr(opt, 'backbone', 'clip-ViT-L/14'))
        # If loading checkpoint or evaluating, do not download or load external backbone weights
        is_loading = getattr(opt, '_loading_checkpoint', False) or getattr(opt, 'continue_train', False)
        pretrained_flag = self.isTrain and not is_loading and getattr(opt, 'pretrained', False)
        weights_path = (getattr(opt, 'drct_backbone_weights', None) or getattr(opt, 'backbone_weights', None)) if pretrained_flag else None
        freeze_encoder = getattr(opt, 'drct_freeze_encoder', True)

        projection_activation = getattr(opt, 'drct_projection_activation', None)
        projection_dropout = float(getattr(opt, 'drct_projection_dropout', 0.0))
        allow_download = bool(getattr(opt, 'drct_allow_download', getattr(opt, 'allow_download', False)))

        # Allow injected encoder module for testing
        encoder_module = getattr(opt, '_injected_encoder', None)
        backbone_dim = getattr(opt, '_injected_backbone_dim', None)

        self.model = DRCT128Model(
            backbone_name=backbone_name,
            pretrained=pretrained_flag,
            weights_path=weights_path,
            freeze_encoder=freeze_encoder,
            embed_dim=embed_dim,
            projection_activation=projection_activation,
            projection_dropout=projection_dropout,
            encoder_module=encoder_module,
            backbone_dim=backbone_dim,
            allow_download=allow_download,
        )

        # Initialize loss function
        alpha = float(getattr(opt, 'drct_alpha', 0.3))
        pos_margin = float(getattr(opt, 'drct_pos_margin', 0.0))
        neg_margin = float(getattr(opt, 'drct_neg_margin', 1.0))
        enable_contrastive = bool(getattr(opt, 'drct_enable_contrastive', True))

        self.loss_fn = DRCTCombinedLoss(
            alpha=alpha,
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            enable_contrastive=enable_contrastive,
        )

        if self.isTrain:
            self._init_optimizer(opt)

        if getattr(opt, 'channels_last', False):
            self.model = self.model.to(memory_format=torch.channels_last)

        self.model.to(self.device)

        # Output and feature placeholders
        self.output: Optional[torch.Tensor] = None
        self.embedding: Optional[torch.Tensor] = None
        self.cls_loss: Optional[torch.Tensor] = None
        self.contrastive_loss: Optional[torch.Tensor] = None

    def _init_optimizer(self, opt: Any) -> None:
        """Initialize optimizer over trainable parameters."""
        trainable_params = list(self.model.trainable_parameters())
        if not trainable_params:
            raise ValueError("[DRCT_128Trainer] No trainable parameters found!")

        lr = float(getattr(opt, 'lr', 1e-4))
        weight_decay = float(getattr(opt, 'weight_decay', 0.0))
        beta1 = float(getattr(opt, 'beta1', 0.9))
        momentum = float(getattr(opt, 'momentum', 0.9))
        optim_name = getattr(opt, 'optim', getattr(opt, 'optimizer', 'adam')).lower()

        if not self.model.freeze_encoder:
            backbone_lr_mult = float(getattr(opt, 'backbone_lr_mult', 0.1))
            encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            head_params = [p for p in self.model.projection.parameters() if p.requires_grad] + \
                          [p for p in self.model.fc.parameters() if p.requires_grad]
            param_groups = [
                {'params': encoder_params, 'lr': lr * backbone_lr_mult},
                {'params': head_params, 'lr': lr},
            ]
        else:
            param_groups = [{'params': trainable_params, 'lr': lr}]

        if optim_name == 'adam':
            self.optimizer = torch.optim.Adam(
                param_groups, betas=(beta1, 0.999), weight_decay=weight_decay
            )
        elif optim_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                param_groups, betas=(beta1, 0.999), weight_decay=weight_decay
            )
        elif optim_name == 'sgd':
            self.optimizer = torch.optim.SGD(
                param_groups, momentum=momentum, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optim_name!r}. Supported: ['adam', 'adamw', 'sgd']")

    def train(self, mode: bool = True) -> "DRCT_128Trainer":
        """Override train to ensure frozen backbone remains in eval mode if frozen."""
        super().train(mode)
        self.model.train(mode)
        if self.model.freeze_encoder:
            self.model.encoder.eval()
        return self

    def eval(self) -> "DRCT_128Trainer":
        """Set model to eval mode."""
        self.train(False)
        return self


    def set_input(self, batch: Any) -> None:
        """Unpack input batch and transfer to target device."""
        non_blocking = (self.device.type == 'cuda' and getattr(self.opt, 'pin_memory', True))
        if isinstance(batch, (tuple, list)):
            inp = batch[0].to(self.device, non_blocking=non_blocking)
            target = batch[1].to(self.device, non_blocking=non_blocking).float()
        elif isinstance(batch, dict):
            inp = batch['image'].to(self.device, non_blocking=non_blocking)
            target = batch['label'].to(self.device, non_blocking=non_blocking).float()
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        if getattr(self.opt, 'channels_last', False):
            inp = inp.to(memory_format=torch.channels_last)

        self.input = inp
        self.label = target

    def forward(self) -> torch.Tensor:
        """Forward pass through model (including DDP wrapper if present) extracting logits and features."""
        self.output, self.embedding = self.model(self.input, return_feature=True)
        return self.output

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract deterministic 128-dimensional embedding in eval mode."""
        underlying = self.model.module if hasattr(self.model, 'module') else self.model
        return underlying.forward_features(x)


    def get_loss(self) -> torch.Tensor:
        """Compute training loss (combined classification + contrastive) or evaluation loss (pure classification)."""
        is_train_mode = bool(self.training and getattr(self.model, 'training', False))
        if is_train_mode:
            total_loss, cls_loss, cont_loss = self.loss_fn(self.output, self.label, self.embedding)
            self.cls_loss = cls_loss.detach()
            self.contrastive_loss = cont_loss.detach()
            self.loss = total_loss
            return total_loss
        else:
            # Ordinary validation uses hard-label classification loss without batch-dependent contrastive loss
            cls_loss = self.loss_fn.cls_criterion(self.output.view(-1), self.label.view(-1).float())
            self.cls_loss = cls_loss.detach()
            self.contrastive_loss = torch.tensor(0.0, device=self.output.device)
            self.loss = cls_loss
            return cls_loss


    def optimize_parameters(self) -> torch.Tensor:
        """Standard optimization step: forward, loss, backward, step."""
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def adjust_learning_rate(self, min_lr: float = 1e-6) -> bool:
        """Decay learning rate across all parameter groups."""
        active = True
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.0
            if param_group['lr'] < min_lr:
                active = False
        return active
