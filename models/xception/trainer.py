"""
XceptionRaw Trainer.

Xception model for binary deepfake classification.
Uses pretrained Xception with Linear(2048, 1) head.

Preserved exactly from xception/trainer_Xception.py.
"""
import torch
import torch.nn as nn
from Refactored.models.base.base_model import BaseModel, init_weights
from Refactored.models.shared.xception_arch import xception


class XceptionRawTrainer(BaseModel):
    """Trainer class for Xception binary classification"""

    def name(self):
        return 'XceptionRaw'

    def __init__(self, opt):
        super(XceptionRawTrainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            # Load pretrained Xception and replace final layer for binary classification
            self.model = xception(pretrained=True)
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            # Create model without pretrained weights (will load from checkpoint)
            self.model = xception(pretrained=False, num_classes=1)

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
