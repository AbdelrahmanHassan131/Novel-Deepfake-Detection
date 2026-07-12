"""
ConvNeXt_128 Trainer.

ConvNeXt-Base with a 128-dim embedding head:
    Linear(1024, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 1)

Follows the same pattern as Wang2020_128 trainer.
"""
import torch
import torch.nn as nn
from torchvision import models as tv_models
from models.base.base_model import BaseModel, init_weights


class ConvNeXt128Trainer(BaseModel):
    """Trainer class for ConvNeXt-Base with 128-dim embedding head."""

    def name(self):
        return 'ConvNeXt_128'

    def __init__(self, opt):
        super(ConvNeXt128Trainer, self).__init__(opt)

        # Determine if we should load pretrained weights
        pretrained_flag = self.isTrain and not opt.continue_train

        # Always create the same architecture
        if pretrained_flag:
            self.model = tv_models.convnext_base(weights=tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            self.model = tv_models.convnext_base(weights=None)

        # Replace the final classifier with a 128-dim embedding head
        # ConvNeXt classifier is: Sequential(LayerNorm, Flatten, Linear(1024, 1000))
        in_features = self.model.classifier[2].in_features  # 1024
        self.model.classifier[2] = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

        # Only initialize weights for brand new training
        if self.isTrain and not opt.continue_train:
            torch.nn.init.normal_(
                self.model.classifier[2][0].weight.data, 0.0, opt.init_gain)
            torch.nn.init.normal_(
                self.model.classifier[2][3].weight.data, 0.0, opt.init_gain)

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
