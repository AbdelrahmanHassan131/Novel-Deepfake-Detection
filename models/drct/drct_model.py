"""DRCT Architecture Definitions: DRCT128Model and DRCTNativeReference.

Implements the official DRCT visual architecture adapted to a 128-dimensional
intermediate embedding, as well as the native-width reference adapter.
"""
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from models.drct.backbone import build_drct_backbone


def build_drct_projection(
    in_dim: int,
    embed_dim: int = 128,
    activation: Optional[str] = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build projection from backbone feature dimension to 128D embedding."""
    proj_layers = [nn.Linear(in_dim, embed_dim)]
    if activation == "relu":
        proj_layers.append(nn.ReLU(inplace=True))
    elif activation == "gelu":
        proj_layers.append(nn.GELU())
    elif activation is not None and activation != "none":
        raise ValueError(f"Unsupported projection_activation: '{activation}'. Use 'none', 'relu', or 'gelu'.")

    if dropout > 0.0:
        proj_layers.append(nn.Dropout(p=dropout))

    return nn.Sequential(*proj_layers)


class DRCT128Model(nn.Module):
    """DRCT model with 128-dimensional intermediate feature representation.
    
    Structure:
        RGB image -> Visual Encoder -> Visual Features (e.g. 768)
                  -> Trainable Linear Projection -> 128D Embedding
                  -> Linear(128, 1) -> Binary logit [B, 1]
    
    The same 128D embedding feeds the standalone binary classifier and MHA fusion.
    """
    def __init__(
        self,
        backbone_name: str = "clip-ViT-L/14",
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        freeze_encoder: bool = True,
        embed_dim: int = 128,
        projection_activation: Optional[str] = None,
        projection_dropout: float = 0.0,
        encoder_module: Optional[nn.Module] = None,
        backbone_dim: Optional[int] = None,
        allow_download: bool = False,
    ):
        super().__init__()
        if embed_dim != 128:
            raise ValueError(f"DRCT_128 strictly requires embed_dim=128, but received embed_dim={embed_dim}.")

        self.embed_dim = embed_dim
        self.freeze_encoder = freeze_encoder
        self.backbone_name = backbone_name

        # 1. Visual Encoder
        if encoder_module is not None:
            self.encoder = encoder_module
            self.backbone_dim = backbone_dim if backbone_dim is not None else 768
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
        else:
            self.encoder, self.backbone_dim = build_drct_backbone(
                name=backbone_name,
                pretrained=pretrained,
                weights_path=weights_path,
                freeze=freeze_encoder,
                allow_download=allow_download,
            )

        # 2. Trainable Projection (visual features -> 128)
        self.projection = build_drct_projection(
            in_dim=self.backbone_dim,
            embed_dim=self.embed_dim,
            activation=projection_activation,
            dropout=projection_dropout,
        )

        # 3. Trainable Binary Classifier (128 -> 1 logit)
        self.fc = nn.Linear(self.embed_dim, 1)

        # Initialize projection and classifier weights
        self._init_head()

    def _init_head(self, init_gain: float = 0.02) -> None:
        """Initialize projection and classifier parameters."""
        for m in (self.projection, self.fc):
            for submodule in m.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.normal_(submodule.weight.data, 0.0, init_gain)
                    if submodule.bias is not None:
                        nn.init.constant_(submodule.bias.data, 0.0)

    def train(self, mode: bool = True) -> "DRCT128Model":
        """Set training mode while keeping frozen encoder in eval mode."""
        super().train(mode)
        if mode and self.freeze_encoder:
            self.encoder.eval()
        return self

    def _extract_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run visual encoder respecting input gradient contract."""
        # If input requires gradient (e.g. adversarial test / input diagnostic),
        # allow autograd to track encoder operations even if encoder params are frozen.
        if self.freeze_encoder and not x.requires_grad:
            with torch.no_grad():
                feat = self.encoder(x)
        else:
            feat = self.encoder(x)
            
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        return feat

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-dimensional embedding.
        
        Preserves caller ambient autograd mode and module training state.
        Computes the pre-dropout representation matching the penultimate feature feeding fc.
        """
        vis_feat = self._extract_encoder_features(x)
        if len(self.projection) > 1 and isinstance(self.projection[-1], nn.Dropout):
            feat = vis_feat
            for layer in list(self.projection.children())[:-1]:
                feat = layer(feat)
        else:
            feat = self.projection(vis_feat)
        return feat

    def forward(
        self, x: torch.Tensor, return_feature: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute forward pass without encoding images twice.
        
        Args:
            x: Input tensor [B, 3, H, W]
            return_feature: If True, returns (logits [B, 1], features_128 [B, 128])
            
        Returns:
            logits [B, 1] or (logits [B, 1], features_128 [B, 128])
        """
        vis_feat = self._extract_encoder_features(x)
        features_128 = self.projection(vis_feat)
        logits = self.fc(features_128)
        if return_feature:
            return logits, features_128
        return logits

    def trainable_parameters(self):
        """Yield trainable parameters (projection + classifier, plus encoder if unfrozen)."""
        if not self.freeze_encoder:
            for param in self.encoder.parameters():
                if param.requires_grad:
                    yield param
        for param in self.projection.parameters():
            if param.requires_grad:
                yield param
        for param in self.fc.parameters():
            if param.requires_grad:
                yield param


class DRCTNativeReference(nn.Module):
    """Native-width reference model matching official DRCT configuration.
    
    Structure:
        RGB image -> Visual Encoder (e.g. 768)
                  -> Native Projection (e.g. 1024)
                  -> Linear(1024, 2) -> Two logits [B, 2] (class 0: real, class 1: fake)
    
    Exposes forward_single_logit returning z = logit_fake - logit_real,
    where sigmoid(z) == softmax(logits)[:, 1].
    """
    def __init__(
        self,
        backbone_name: str = "clip-ViT-L/14",
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        freeze_encoder: bool = False,
        embedding_size: int = 1024,
        encoder_module: Optional[nn.Module] = None,
        backbone_dim: Optional[int] = None,
        allow_download: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.freeze_encoder = freeze_encoder
        self.backbone_name = backbone_name

        if encoder_module is not None:
            self.encoder = encoder_module
            self.backbone_dim = backbone_dim if backbone_dim is not None else 768
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
        else:
            self.encoder, self.backbone_dim = build_drct_backbone(
                name=backbone_name,
                pretrained=pretrained,
                weights_path=weights_path,
                freeze=freeze_encoder,
                allow_download=allow_download,
            )

        self.projection = nn.Linear(self.backbone_dim, self.embedding_size)
        self.fc = nn.Linear(self.embedding_size, 2)

    def forward(
        self, x: torch.Tensor, return_feature: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        vis_feat = self.encoder(x)
        if isinstance(vis_feat, (tuple, list)):
            vis_feat = vis_feat[0]
        feature = self.projection(vis_feat)
        logits = self.fc(feature)
        if return_feature:
            return logits, feature
        return logits

    def forward_single_logit(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw single logit z = logit_fake - logit_real [B, 1].
        
        Proof:
            softmax([s0, s1])_1 = exp(s1) / (exp(s0) + exp(s1))
                                = 1 / (1 + exp(-(s1 - s0)))
                                = sigmoid(s1 - s0).
        """
        logits = self.forward(x, return_feature=False)  # [B, 2]
        return (logits[:, 1:2] - logits[:, 0:1])
