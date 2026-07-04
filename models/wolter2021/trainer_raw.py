"""
WolterWavelet2021Raw Trainer.

Original Wolter et al. 2022 architecture without modifications.
Uses WaveletPacketCNN with 512 -> 1 classifier (no embedding layer).

Preserved exactly from WaveletsRawModelRawInput/trainer_WaveletsRawModelRawInput.py.
"""
import torch
import torch.nn as nn
from Refactored.models.base.base_model import BaseModel, init_weights
from .wavelet_cnn import WaveletPacketCNN
from .wavelet_utils import compute_wavelet_packet_coeffs, log_scale_packets


class WolterWaveletRawTrainer(BaseModel):
    """
    Wavelet-Packet DeepFake Detection Trainer
    Based on: Wolter et al. "Wavelet-Packets for Deepfake Image Analysis and Detection"
    Machine Learning, ECML PKDD 2022 Journal Track

    Original architecture without modifications
    """

    def name(self):
        return 'WolterWavelet2021Raw'

    def __init__(self, opt):
        super(WolterWaveletRawTrainer, self).__init__(opt)

        # Wavelet packet parameters
        self.wavelet_type = getattr(opt, 'wavelet_type', 'haar')
        self.wavelet_level = getattr(opt, 'wavelet_level', 3)
        self.wavelet_mode = getattr(opt, 'wavelet_mode', 'reflect')
        self.use_log_packets = getattr(opt, 'use_log_packets', True)

        # Calculate number of input channels
        # At level 3: 4^3 = 64 packets per channel
        # For RGB: 3 * 64 = 192 channels
        num_packets_per_channel = 4 ** self.wavelet_level
        input_channels = 3 * num_packets_per_channel

        # Create model
        if self.isTrain and not opt.continue_train:
            self.model = WaveletPacketCNN(
                input_channels=input_channels,
                num_classes=1
            )
            init_weights(self.model, init_type='normal', gain=opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = WaveletPacketCNN(
                input_channels=input_channels,
                num_classes=1
            )

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()

            # Initialize optimizer
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

        # Load checkpoint if needed
        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model.to(opt.gpu_ids[0])

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def __call__(self, input_tensor):
        """For validation - accepts wavelet packets from dataloader"""
        if input_tensor.shape[1] == 192:  # Already wavelet packets
            output = self.model(input_tensor.to(self.device))
            return output

        # Fallback: compute wavelets if RGB input
        batch_packets = []
        for img in input_tensor:
            packets = compute_wavelet_packet_coeffs(
                img, self.wavelet_type, self.wavelet_level, self.wavelet_mode
            )
            if self.use_log_packets:
                packets = log_scale_packets(packets)
            batch_packets.append(packets)

        wavelet_input = torch.stack(batch_packets).to(self.device)
        output = self.model(wavelet_input)
        return output

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def set_input(self, input):
        """Input from dataloader: (wavelet_packets, labels)"""
        wavelet_packets, labels = input[0], input[1]
        self.input = wavelet_packets.to(self.device)
        self.label = labels.to(self.device).float()

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
