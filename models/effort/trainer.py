"""EFFORT_128 Model Trainer integrating EFFORT128Model with the core training engine."""
from typing import Any, Optional
import torch
from models.base.base_model import BaseModel
from models.effort.effort_model import EFFORT128Model
from models.effort.loss import EFFORTCombinedLoss


class EFFORT_128Trainer(BaseModel):
    """Trainer for EFFORT_128 (Multi-Layer + Orthogonal SVD CLIP detector with 128-D embedding)."""

    def name(self) -> str:
        return "EFFORT_128"

    def __init__(self, opt: Any):
        super(EFFORT_128Trainer, self).__init__(opt)

        embed_dim = int(getattr(opt, "embed_dim", 128))
        if embed_dim != 128:
            raise ValueError(f"EFFORT_128 requires embed_dim=128, got embed_dim={embed_dim}.")

        backbone_name = getattr(opt, "drct_backbone", getattr(opt, "backbone", "clip-ViT-L/14"))
        is_loading = getattr(opt, "_loading_checkpoint", False) or getattr(opt, "continue_train", False)
        pretrained_flag = self.isTrain and not is_loading and getattr(opt, "pretrained", False)
        weights_path = (
            getattr(opt, "drct_backbone_weights", None) or getattr(opt, "backbone_weights", None)
        ) if pretrained_flag else None
        allow_download = bool(getattr(opt, "drct_allow_download", getattr(opt, "allow_download", False)))

        svd_rank = int(getattr(opt, "effort_svd_rank", 16))
        svd_last_n_blocks = int(getattr(opt, "effort_svd_blocks", 12))
        multi_layer = bool(getattr(opt, "effort_multi_layer", True))
        num_tapped_layers = int(getattr(opt, "effort_num_layers", 4))
        head_hidden_dim = int(getattr(opt, "effort_head_hidden", 256))
        head_dropout = float(getattr(opt, "effort_dropout", 0.3))
        layer_indices = getattr(opt, "effort_layer_indices", None)
        if isinstance(layer_indices, str) and layer_indices.strip():
            layer_indices = [int(x.strip()) for x in layer_indices.split(",") if x.strip()]
        elif not layer_indices:
            layer_indices = None

        encoder_module = getattr(opt, "_injected_encoder", None)
        backbone_dim = getattr(opt, "_injected_backbone_dim", None)

        self.model = EFFORT128Model(
            backbone_name=backbone_name,
            pretrained=pretrained_flag,
            weights_path=weights_path,
            allow_download=allow_download,
            embed_dim=embed_dim,
            svd_rank=svd_rank,
            svd_last_n_blocks=svd_last_n_blocks,
            multi_layer=multi_layer,
            num_tapped_layers=num_tapped_layers,
            layer_indices=layer_indices,
            head_hidden_dim=head_hidden_dim,
            head_dropout=head_dropout,
            encoder_module=encoder_module,
            backbone_dim=backbone_dim,
        )

        # Ensure preprocessing version is recorded in checkpoint options
        if not hasattr(opt, "effort_preprocessing_version") or not getattr(opt, "effort_preprocessing_version", None):
            opt.effort_preprocessing_version = "clip_forensic_v1"

        label_smoothing = float(getattr(opt, "effort_label_smoothing", 0.05))
        contrastive_alpha = float(getattr(opt, "drct_alpha", 0.0))
        pos_margin = float(getattr(opt, "drct_pos_margin", 0.0))
        neg_margin = float(getattr(opt, "drct_neg_margin", 1.0))
        ortho_weight = float(getattr(opt, "effort_ortho_weight", 0.1))

        self.loss_fn = EFFORTCombinedLoss(
            label_smoothing=label_smoothing,
            contrastive_alpha=contrastive_alpha,
            pos_margin=pos_margin,
            neg_margin=neg_margin,
            ortho_weight=ortho_weight,
        )

        if self.isTrain:
            self._init_optimizer(opt)

        if getattr(opt, "channels_last", False):
            self.model = self.model.to(memory_format=torch.channels_last)

        self.model.to(self.device)

        self.output: Optional[torch.Tensor] = None
        self.embedding: Optional[torch.Tensor] = None
        self.cls_loss: Optional[torch.Tensor] = None
        self.contrastive_loss: Optional[torch.Tensor] = None
        self.ortho_loss: Optional[torch.Tensor] = None

    def _init_optimizer(self, opt: Any) -> None:
        lr = float(getattr(opt, "lr", 1e-4))
        weight_decay = float(getattr(opt, "weight_decay", 1e-2))
        beta1 = float(getattr(opt, "beta1", 0.9))
        momentum = float(getattr(opt, "momentum", 0.9))
        backbone_lr_mult = float(getattr(opt, "backbone_lr_mult", 0.2))
        optim_name = getattr(opt, "optim", getattr(opt, "optimizer", "adamw")).lower()

        param_groups = self.model.get_param_groups(
            lr=lr,
            backbone_lr_mult=backbone_lr_mult,
            weight_decay=weight_decay,
        )
        if not param_groups:
            raise ValueError("[EFFORT_128Trainer] No trainable parameters found!")

        if optim_name == "adamw":
            self.optimizer = torch.optim.AdamW(param_groups, betas=(beta1, 0.999))
        elif optim_name == "adam":
            self.optimizer = torch.optim.Adam(param_groups, betas=(beta1, 0.999))
        elif optim_name == "sgd":
            self.optimizer = torch.optim.SGD(param_groups, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {optim_name!r}. Supported: ['adamw', 'adam', 'sgd']")

    def train(self, mode: bool = True) -> "EFFORT_128Trainer":
        super().train(mode)
        self.model.train(mode)
        return self

    def eval(self) -> "EFFORT_128Trainer":
        self.train(False)
        return self

    def set_input(self, batch: Any) -> None:
        non_blocking = self.device.type == "cuda" and getattr(self.opt, "pin_memory", True)
        if isinstance(batch, (tuple, list)):
            inp = batch[0].to(self.device, non_blocking=non_blocking)
            target = batch[1].to(self.device, non_blocking=non_blocking).float()
        elif isinstance(batch, dict):
            inp = batch["image"].to(self.device, non_blocking=non_blocking)
            target = batch["label"].to(self.device, non_blocking=non_blocking).float()
        else:
            raise TypeError(f"Unexpected batch type: {type(batch)}")

        if getattr(self.opt, "channels_last", False):
            inp = inp.to(memory_format=torch.channels_last)

        self.input = inp
        self.label = target

    def forward(self) -> torch.Tensor:
        self.output, self.embedding = self.model(self.input, return_feature=True)
        return self.output

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        underlying = self.model.module if hasattr(self.model, "module") else self.model
        return underlying.forward_features(x)

    def get_loss(self) -> torch.Tensor:
        is_train_mode = bool(self.training and getattr(self.model, "training", False))
        underlying = self.model.module if hasattr(self.model, "module") else self.model
        ortho = underlying.orthogonality_loss() if is_train_mode else None

        total_loss, cls_loss, cont_loss, o_loss = self.loss_fn(
            self.output,
            self.label,
            embeddings=self.embedding,
            ortho_loss=ortho,
            is_train=is_train_mode,
        )
        self.cls_loss = cls_loss.detach()
        self.contrastive_loss = cont_loss.detach()
        self.ortho_loss = o_loss.detach()
        self.loss = total_loss
        return total_loss

    def optimize_parameters(self) -> torch.Tensor:
        self.forward()
        self.loss = self.get_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def adjust_learning_rate(self, min_lr: float = 1e-6) -> bool:
        active = True
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                active = False
        return active
