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
        pretrained_flag = self.isTrain and not opt.continue_train

        # Always create the same architecture!
        self.model = resnet50(pretrained=pretrained_flag)
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
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])

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
