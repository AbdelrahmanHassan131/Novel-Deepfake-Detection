"""
Xception_128 Trainer.

Xception model with a 128-dim embedding head:
    Linear(2048, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 1)

Mirrors the Wang2020_128 pattern applied to the Xception backbone.
"""
import torch
import torch.nn as nn
from models.base.base_model import BaseModel, init_weights
from models.shared.xception_arch import xception


class Xception128Trainer(BaseModel):
    """Trainer class for Xception with 128-D embedding head."""

    def name(self):
        return 'Xception_128'

    def __init__(self, opt):
        super(Xception128Trainer, self).__init__(opt)

        # Determine if we should load pretrained weights
        pretrained_flag = self.isTrain and not opt.continue_train

        # Always create the same architecture!
        self.model = xception(pretrained=pretrained_flag)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        # Only initialize weights for brand new training
        if self.isTrain and not opt.continue_train:
            torch.nn.init.normal_(
                self.model.fc[0].weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(
                self.model.fc[3].weight.data, 0.0, opt.init_gain)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # Initialize optimizer
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999)
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

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

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

    def train(self):
        self.model.train()
