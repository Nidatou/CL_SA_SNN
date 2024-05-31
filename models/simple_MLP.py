import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

from models.utils import loss_func
from models.cl.continual_learner import CLBase
from models.fc.nets import SDM_Base


class Simple(CLBase):
    def __init__(
        self
    ):
        super(Simple, self).__init__()

        self.flatten = nn.Flatten()
        self.sdm_nets = nn.Sequential(
            nn.Linear(784, 400, bias=False),
            nn.ReLU(),
            nn.Linear(400, 10, bias=False),
        )

    @property
    def name(self):
        return "what"
        # return self.sdm_nets.name

    def enforce_weights_regulation(self):
        # self.sdm_nets.enforce_weight_regularization()
        return

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.flatten(x)
        x = self.sdm_nets(x)

        return x

    def classify(self, x: torch.Tensor):
        x = self(x)
        return x

    def train_a_batch(
        self, optimizer: torch.optim.Optimizer, x, y=None, x_=None, y_=None, scores_=None, rnt=0.5,
        active_classes=None, task=1, replay_not_hidden=False, freeze_convE=False, **kwargs
    ):
        self.train()

        optimizer.zero_grad()

        y_hat = self(x)
        if active_classes is not None:
            # for Task-IL and class-IL, model need to remove several output
            # and task-IL only keep the classes for current task
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            y_hat = y_hat[:, class_entries]
        if y is not None and len(y.size()) == 0:
            y = y.expand(1)
        predL = F.cross_entropy(input=y_hat, target=y, reduction='none')
        predL = torch.mean(predL, dim=0)

        # Weigh losses
        loss_cur = predL
        # Calculate training-accuracy
        accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

        predL.backward()
        optimizer.step()

        return {
            'loss_total': predL,
            'loss_current': loss_cur,
            'loss_replay': 0.0,
            'pred': predL,
            'pred_re': 0.0,
            'distil_re': 0.0,
            'accuracy': accuracy
        }

