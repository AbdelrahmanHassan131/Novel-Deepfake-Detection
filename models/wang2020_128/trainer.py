"""
Wang2020_128 Trainer.

ResNet-50 with a 128-dim embedding head:
    Linear(2048, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 1)

Preserved exactly from MyModels/networks/wang2020_128/Trainer_Wang2020_128.py.
"""
import functools
import torch
import torch.nn as nn
from models.shared.resnet import resnet50
from models.base.base_model import BaseModel, init_weights


class Wang2020_128Trainer(BaseModel):
    def name(self):
        return 'Wang2020_128'

    def __init__(self, opt):
        super(Wang2020_128Trainer, self).__init__(opt)

        # Determine if we should load pretrained weights
        pretrained_flag = self.isTrain and not opt.continue_train and getattr(opt, 'pretrained', True)

        # Always create the same architecture!
        weights_path = getattr(opt, 'backbone_weights', None)
        self.model = resnet50(pretrained=pretrained_flag, weights_path=weights_path)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, getattr(opt, 'embed_dim', 128)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(getattr(opt, 'embed_dim', 128), 1)
        )

        # Only initialize weights for brand new training
        if self.isTrain and not opt.continue_train:
            torch.nn.init.normal_(
                self.model.fc[0].weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(
                self.model.fc[3].weight.data, 0.0, opt.init_gain)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            weight_decay = getattr(opt, 'weight_decay', 0.0)
            momentum = getattr(opt, 'momentum', 0.0)
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),
                    weight_decay=weight_decay,
                )
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=opt.lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                )
            else:
                raise ValueError("optim should be [adam, sgd]")

        # Only call legacy load_networks if explicitly in evaluation mode,
        # never during training resume which is handled centrally by BaseTrainer.
        if not self.isTrain:
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
