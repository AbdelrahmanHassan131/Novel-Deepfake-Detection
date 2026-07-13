"""
Concatenation Fusion WWXC Trainer (Fusion_WWXC).

Combines 4 pre-trained 128-D models:
  - Wang2020_128  (ResNet-50, 128-D embedding)
  - Wolter_128    (WaveletPacketCNN, 128-D embedding)
  - Xception_128  (Xception, 128-D embedding)
  - ConvNeXt_128  (ConvNeXt-Base, 128-D embedding)

using simple concatenation and MLP fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base.base_model import BaseModel, init_weights
from .fusion_wwxc_classifier import ConcatenationFusionWWXCClassifier


class ConcatenationFusionWWXCTrainer(BaseModel):
    """
    Trainer for 4-model Concatenation Fusion (WWXC).
    Combines Wang2020_128 + Wolter_128 + Xception_128 + ConvNeXt_128.
    """

    def name(self):
        return 'Fusion_WWXC'

    def __init__(self, opt):
        super(ConcatenationFusionWWXCTrainer, self).__init__(opt)

        # Fusion parameters
        self.embed_dim = getattr(opt, 'embed_dim', 128)
        self.dropout = getattr(opt, 'dropout', 0.1)
        self.freeze_base_models = getattr(opt, 'freeze_base_models', True)

        # Paths to pre-trained models
        self.rgb_model_path = opt.rgb_model_path
        self.wavelet_model_path = opt.wavelet_model_path
        self.xception_model_path = opt.xception_model_path
        self.convnext_model_path = opt.convnext_model_path

        # Load all 4 pre-trained base models
        print(f"Loading Wang2020_128 model from: {self.rgb_model_path}")
        self.rgb_model = self._load_rgb_model(opt)

        print(f"Loading Wolter_128 model from: {self.wavelet_model_path}")
        self.wavelet_model = self._load_wavelet_model(opt)

        print(f"Loading Xception_128 model from: {self.xception_model_path}")
        self.xception_model = self._load_xception_model(opt)

        print(f"Loading ConvNeXt_128 model from: {self.convnext_model_path}")
        self.convnext_model = self._load_convnext_model(opt)

        # Freeze base models if specified
        if self.freeze_base_models:
            print("Freezing all 4 base models")
            self._freeze_model(self.rgb_model)
            self._freeze_model(self.wavelet_model)
            self._freeze_model(self.xception_model)
            self._freeze_model(self.convnext_model)

        # Create fusion model
        if self.isTrain and not opt.continue_train:
            self.model = ConcatenationFusionWWXCClassifier(
                embed_dim=self.embed_dim,
                dropout=self.dropout
            )
            init_weights(self.model, init_type='xavier', gain=opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = ConcatenationFusionWWXCClassifier(
                embed_dim=self.embed_dim,
                dropout=self.dropout
            )

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            # Optimizer for fusion model only (base models frozen)
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),
                    weight_decay=1e-4
                )
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=opt.lr,
                    momentum=0.9,
                    weight_decay=1e-4
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        # Load checkpoint if continuing training or evaluating
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        # Move all models to device
        self.rgb_model.to(self.device)
        self.wavelet_model.to(self.device)
        self.xception_model.to(self.device)
        self.convnext_model.to(self.device)
        self.model.to(self.device)

        print(f"Fusion_WWXC model created (4-model concatenation fusion)")
        print(f"Base models frozen: {self.freeze_base_models}")

    # ------------------------------------------------------------------
    #  Checkpoint helpers
    # ------------------------------------------------------------------
    def _get_weights_dict(self, state_dict):
        """Extract model weights from different checkpoint dictionary formats."""
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                return state_dict['model_state_dict']
            elif 'model' in state_dict:
                return state_dict['model']
            elif 'state_dict' in state_dict:
                return state_dict['state_dict']
        return state_dict

    # ------------------------------------------------------------------
    #  Base model loaders
    # ------------------------------------------------------------------
    def _load_rgb_model(self, opt):
        """Load pre-trained Wang2020_128 (ResNet-50, 128-D embedding)."""
        from models.shared.resnet import resnet50

        model = resnet50(num_classes=1)
        state_dict = torch.load(self.rgb_model_path, map_location='cuda')
        weights = self._get_weights_dict(state_dict)

        # Match saved architecture if fc is Sequential
        if 'fc.0.weight' in weights:
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )

        model.load_state_dict(weights)

        # Strip final classifier to get 128-dim embeddings
        model.fc = nn.Sequential(
            model.fc[0],  # Linear(2048 -> 128)
            model.fc[1],  # ReLU
        )

        model.eval()
        return model

    def _load_wavelet_model(self, opt):
        """Load pre-trained Wolter_128 (WaveletPacketCNN, 128-D embedding)."""
        from models.wolter2021.wavelet_cnn import WaveletPacketCNN128

        wavelet_level = getattr(opt, 'wavelet_level', 3)
        num_packets_per_channel = 4 ** wavelet_level
        input_channels = 3 * num_packets_per_channel

        model = WaveletPacketCNN128(input_channels=input_channels, num_classes=1)
        state_dict = torch.load(self.wavelet_model_path, map_location='cuda')
        weights = self._get_weights_dict(state_dict)
        model.load_state_dict(weights)

        # Strip final classifier to get 128-dim embeddings
        if isinstance(model.classifier, nn.Sequential):
            model.classifier = nn.Sequential(
                model.classifier[0],  # Linear(512, 128)
                model.classifier[1],  # ReLU
            )
        else:
            raise ValueError(
                "Unexpected classifier structure in WaveletPacketCNN128")

        model.eval()
        return model

    def _load_xception_model(self, opt):
        """Load pre-trained Xception_128 (Xception, 128-D embedding)."""
        from models.shared.xception_arch import xception

        model = xception(pretrained=False, num_classes=1)
        state_dict = torch.load(self.xception_model_path, map_location='cuda')
        weights = self._get_weights_dict(state_dict)

        # Match saved architecture if fc is Sequential
        if 'fc.0.weight' in weights:
            model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )

        model.load_state_dict(weights)

        # Strip final classifier to get 128-dim embeddings
        model.fc = nn.Sequential(
            model.fc[0],  # Linear(2048 -> 128)
            model.fc[1],  # ReLU
        )

        model.eval()
        return model

    def _load_convnext_model(self, opt):
        """Load pre-trained ConvNeXt_128 (ConvNeXt-Base, 128-D embedding)."""
        from torchvision import models as tv_models

        model = tv_models.convnext_base(weights=None)

        # Rebuild the 128-D head to match saved checkpoint
        in_features = model.classifier[2].in_features  # 1024
        model.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        state_dict = torch.load(self.convnext_model_path, map_location='cuda')
        weights = self._get_weights_dict(state_dict)
        model.load_state_dict(weights)

        # Strip final classifier to get 128-dim embeddings
        model.classifier[2] = nn.Sequential(
            model.classifier[2][0],  # Linear(1024 -> 128)
            model.classifier[2][1],  # ReLU
        )

        model.eval()
        return model

    def _freeze_model(self, model):
        """Freeze all parameters of a model."""
        for param in model.parameters():
            param.requires_grad = False

    def _get_gpu_wavelet_backend(self):
        """Lazy-init the GPU wavelet backend on first use."""
        if not hasattr(self, '_gpu_wavelet') or self._gpu_wavelet is None:
            from data.wavelets.backends.gpu_backend import GPUWaveletBackend
            wavelet_level = getattr(self.opt, 'wavelet_level', 3)
            wavelet_type = getattr(self.opt, 'wavelet_type', 'haar')
            wavelet_mode = getattr(self.opt, 'wavelet_mode', 'reflect')
            use_log = getattr(self.opt, 'use_log_packets', True)
            self._gpu_wavelet = GPUWaveletBackend(
                wavelet=wavelet_type,
                level=wavelet_level,
                mode=wavelet_mode,
                log_scale=use_log,
                device=self.device,
            )
        return self._gpu_wavelet

    # ------------------------------------------------------------------
    #  Training interface
    # ------------------------------------------------------------------
    def adjust_learning_rate(self, min_lr=1e-6):
        """Reduce learning rate by a factor of 10."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def __call__(self, rgb_input, wavelet_input):
        """
        Make trainer callable for validation.
        All 4 base models receive the same RGB input except Wolter
        which uses wavelet_input.
        """
        rgb_input = rgb_input.to(self.device)
        if wavelet_input.shape[1] == 3:
            backend = self._get_gpu_wavelet_backend()
            wavelet_input = backend(wavelet_input.to(self.device))
        else:
            wavelet_input = wavelet_input.to(self.device)

        with torch.no_grad():
            wang_embed = self.rgb_model(rgb_input)           # [B, 128]
            wolter_embed = self.wavelet_model(wavelet_input)  # [B, 128]
            xception_embed = self.xception_model(rgb_input)   # [B, 128]
            convnext_embed = self.convnext_model(rgb_input)   # [B, 128]

        output = self.model(wang_embed, wolter_embed, xception_embed, convnext_embed)
        return output

    def train(self):
        """Set model to training mode."""
        self.model.train()
        if self.freeze_base_models:
            self.rgb_model.eval()
            self.wavelet_model.eval()
            self.xception_model.eval()
            self.convnext_model.eval()

    def eval(self):
        """Set all models to evaluation mode."""
        self.model.eval()
        self.rgb_model.eval()
        self.wavelet_model.eval()
        self.xception_model.eval()
        self.convnext_model.eval()

    def set_input(self, input):
        """
        Process input from dataloader: (rgb_images, wavelet_packets/rgb, labels)

        Wang2020, Xception, ConvNeXt all consume the RGB images.
        Wolter consumes the wavelet packets.
        """
        rgb_imgs, wavelet_imgs, labels = input[0], input[1], input[2]

        self.rgb_input = rgb_imgs.to(self.device)
        self.label = labels.to(self.device).float()

        if wavelet_imgs.shape[1] == 3:
            backend = self._get_gpu_wavelet_backend()
            with torch.no_grad():
                self.wavelet_input = backend(wavelet_imgs.to(self.device))
        else:
            self.wavelet_input = wavelet_imgs.to(self.device)

    def forward(self):
        """Forward pass through all models."""
        with torch.no_grad():
            self.wang_embed = self.rgb_model(self.rgb_input)
            self.wolter_embed = self.wavelet_model(self.wavelet_input)
            self.xception_embed = self.xception_model(self.rgb_input)
            self.convnext_embed = self.convnext_model(self.rgb_input)

        self.output = self.model(
            self.wang_embed, self.wolter_embed,
            self.xception_embed, self.convnext_embed
        )

    def get_loss(self):
        """Calculate and return the loss."""
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        """Perform one optimization step."""
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
