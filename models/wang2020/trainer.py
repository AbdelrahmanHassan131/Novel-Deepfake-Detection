"""
Wang2020 Raw Resolution Trainer.

Original Wang et al. 2020 binary classifier using ResNet-50 with a single
Linear(2048, 1) head. Operates on full-resolution images.

Preserved exactly from MyModels/networks/trainer.py.
"""
import functools
import torch
import torch.nn as nn
from models.shared.resnet import resnet50
from models.base.base_model import BaseModel, init_weights


class Wang2020RawTrainer(BaseModel):
    def name(self):
        return 'Wang2020Raw'

    def __init__(self, opt):
        super(Wang2020RawTrainer, self).__init__(opt)

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=getattr(opt, 'pretrained', True))
            self.model.fc = nn.Linear(2048, 1)
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=1)

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            # initialize optimizers
            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999),
                                                  weight_decay=getattr(opt, 'weight_decay', 0.0))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=getattr(opt, 'momentum', 0.0), weight_decay=getattr(opt, 'weight_decay', 0.0))
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain:
            self.load_networks(opt.epoch)
        if getattr(opt, 'channels_last', False):
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        non_blocking = (self.device.type == 'cuda' and getattr(self.opt, 'pin_memory', True))
        inp = input[0].to(self.device, non_blocking=non_blocking)
        if getattr(self.opt, 'channels_last', False):
            inp = inp.to(memory_format=torch.channels_last)
        self.input = inp
        self.label = input[1].to(self.device, non_blocking=non_blocking).float()

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
