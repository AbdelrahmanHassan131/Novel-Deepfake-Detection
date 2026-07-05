"""
WolterWavelet2021_128 Trainer.

Wavelet Packet CNN with 128-dim embedding head:
    Linear(512, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 1)

Preserved exactly from MyModels/networks/FrequencyModels/WaveletsPacketsScratch_128/
Trainer_WaveletsPacketsScratch_128.py.
"""
import torch
import torch.nn as nn
from models.base.base_model import BaseModel, init_weights
from .wavelet_cnn import WaveletPacketCNN128
from .wavelet_utils import compute_wavelet_packet_coeffs, log_scale_packets


class WolterWavelet128Trainer(BaseModel):
    """
    Wavelet-Packet DeepFake Detection Trainer
    Based on: Wolter et al. "Wavelet-Packets for Deepfake Image Analysis and Detection"
    Machine Learning, ECML PKDD 2022 Journal Track

    OPTIMIZED VERSION: Wavelets are computed in DataLoader workers (parallel CPU processing)
    """

    def name(self):
        return 'WolterWavelet2021_128'

    def __init__(self, opt):
        super(WolterWavelet128Trainer, self).__init__(opt)

        # Wavelet packet parameters (for validation/inference only)
        self.wavelet_type = getattr(opt, 'wavelet_type', 'haar')
        self.wavelet_level = getattr(opt, 'wavelet_level', 3)
        self.wavelet_mode = getattr(opt, 'wavelet_mode', 'reflect')
        self.use_log_packets = getattr(opt, 'use_log_packets', True)
        self._wavelet_backend_name = getattr(opt, 'wavelet_backend', 'cpu')
        self._gpu_wavelet = None  # lazy-init

        # Calculate number of input channels
        # At level 3: 4^3 = 64 packets per channel
        # For RGB: 3 * 64 = 192 channels
        num_packets_per_channel = 4 ** self.wavelet_level
        input_channels = 3 * num_packets_per_channel

        # Determine if we should initialize from scratch or load
        if self.isTrain and not opt.continue_train:
            # Create model from scratch
            self.model = WaveletPacketCNN128(
                input_channels=input_channels,
                num_classes=1
            )
            # Initialize weights using specified initialization
            init_weights(self.model, init_type='normal', gain=opt.init_gain)

        if not self.isTrain or opt.continue_train:
            # Create model architecture (will load weights later)
            self.model = WaveletPacketCNN128(
                input_channels=input_channels,
                num_classes=1
            )

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            # Initialize optimizer with weight decay as in the paper
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),
                    weight_decay=1e-4  # Regularization
                )
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=opt.lr,
                    momentum=0.9,
                    weight_decay=1e-4  # Regularization
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        # Load checkpoint if continuing training or evaluating
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model.to(opt.gpu_ids[0])

    def _get_gpu_wavelet_backend(self):
        """Lazy-init the GPU wavelet backend on first use."""
        if self._gpu_wavelet is None:
            from data.wavelets.backends.gpu_backend import GPUWaveletBackend
            self._gpu_wavelet = GPUWaveletBackend(
                wavelet=self.wavelet_type,
                level=self.wavelet_level,
                mode=self.wavelet_mode,
                log_scale=self.use_log_packets,
                device=self.device,
            )
        return self._gpu_wavelet

    def adjust_learning_rate(self, min_lr=1e-6):
        """Reduce learning rate by a factor of 10"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def __call__(self, input_tensor):
        """
        Make trainer callable for validation.
        """
        if input_tensor.shape[1] == 192:  # Already wavelet packets from dataloader
            output = self.model(input_tensor.to(self.device))
            return output

        # Fallback: Input is RGB, compute wavelets
        if self._wavelet_backend_name == 'gpu':
            backend = self._get_gpu_wavelet_backend()
            wavelet_input = backend(input_tensor.to(self.device))
        else:
            batch_packets = []
            for img in input_tensor:
                packets = compute_wavelet_packet_coeffs(
                    img,
                    wavelet=self.wavelet_type,
                    level=self.wavelet_level,
                    mode=self.wavelet_mode
                )
                if self.use_log_packets:
                    packets = log_scale_packets(packets)
                batch_packets.append(packets)
            wavelet_input = torch.stack(batch_packets).to(self.device)

        output = self.model(wavelet_input)
        return output

    def train(self):
        """Set model to training mode"""
        self.model.train()

    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()

    def set_input(self, input):
        """
        Process input from dataloader: (tensor, labels).

        If tensor has 3 channels (RGB), computes wavelet packets on GPU
        for the entire batch. If 192 channels, uses them directly.
        """
        data, labels = input[0], input[1]
        data = data.to(self.device)
        self.label = labels.to(self.device).float()

        if data.shape[1] == 3:
            # RGB input — compute wavelets on GPU in batch
            backend = self._get_gpu_wavelet_backend()
            with torch.no_grad():
                self.input = backend(data)
        else:
            # Already wavelet packets (192 channels)
            self.input = data

    def forward(self):
        """Forward pass through the model"""
        self.output = self.model(self.input)

    def get_loss(self):
        """Calculate and return the loss"""
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        """Perform one optimization step"""
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
