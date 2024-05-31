from torch import nn
import torch
import random
import numpy as np
from sklearn.preprocessing import minmax_scale

from models.cl.continual_learner import CLBase
from models.conv.nets import SNN_ConvLayer, VGGSNNwoAP
from spikingjelly.activation_based import neuron, layer, surrogate
from utils.utils import minmax_val


class FlyModel_Whole(CLBase):
    def __init__(
        self, image_size, image_channels, output_size, device=None,
        # the parameters for fly-model construction
        n_kc=1000, kc_response=32, k_num=32, fly_lr=5e-3, min_max=True
    ):
        assert device is not None
        super(FlyModel_Whole, self).__init__()

        ########################################################
        # model construction part ##############################
        self.conv_out_units = image_channels * image_size ** 2
        # Fly model part #############
        self.n_kc = n_kc
        self.classes = output_size
        self.label = "Flymodel_whole"
        self.kc_reponse = kc_response  # the number of pns summed up by every kc cell
        self.fly_lr = fly_lr
        self.min_max = min_max

        self.kc_mat = torch.zeros((n_kc, self.conv_out_units)).to(device)
        self.out_mat = torch.zeros(self.classes, n_kc).to(device)
        self.k_num = k_num
        for i in range(n_kc):
            self.kc_mat[i, random.sample(list(range(self.conv_out_units)), self.kc_reponse)] = 1
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.flatten(x)
        if self.min_max:
            x = minmax_val(x, dim=-1)
        with torch.no_grad():
            KCs = self.Generate_KCs(x)
            out = torch.matmul(KCs, self.out_mat.T)

        return out

    def classify(self, x: torch.Tensor, **kwargs):
        x = self.flatten(x)
        if self.min_max:
            x = minmax_val(x, dim=-1)
        with torch.no_grad():
            KCs = self.Generate_KCs(x)
            out = torch.matmul(KCs, self.out_mat.T)

        return out

    def Generate_KCs(self, x: torch.Tensor):
        with torch.no_grad():
            KCs = torch.matmul(x, self.kc_mat.T)
            for i in range(KCs.shape[0]):
                sorted_kc, _ = torch.sort(KCs[i], descending=True)
                threshold = sorted_kc[self.k_num]
                KCs[i][KCs[i] <= threshold] = 0
                # KCs[i][KCs[i] > 0] = 1
                KCs[i] = KCs[i] / torch.max(KCs[i])
        return KCs

    def _device(self):
        return self.kc_mat.data.device

    def _is_on_cuda(self):
        return self.kc_mat.data.is_cuda

    @property
    def name(self):
        return f"{self.label}_pn-{self.conv_out_units}({self.kc_reponse})_kc-{self.n_kc}({self.k_num}){'-mm' if self.min_max else ''}"

    @property
    def classifier_weight(self):
        return self.out_mat.detach().cpu().numpy()

    def train_a_batch(
        self, optimizer: torch.optim.Optimizer, x, y=None, x_=None, y_=None, scores_=None, rnt=0.5,
        active_classes=None, task=1, replay_not_hidden=False, freeze_convE=False, scenario='class', **kwargs
    ):
        # 然而在这里optimser完全没有卵用...
        assert y is not None
        x = self.flatten(x)
        if self.min_max:
            x = minmax_val(x, dim=-1)
        KCs = self.Generate_KCs(x)
        y_hat = torch.matmul(KCs, self.out_mat.T)
        accuracy = (y == y_hat.max(1)[1]).sum().item() / y.shape[0]

        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
        else:
            class_entries = list(range(self.classes))
        for i in class_entries:
            sum_kc = torch.sum(KCs[y+class_entries[0] == i], dim=0)
            self.out_mat[i] += self.fly_lr * sum_kc
        self.out_mat[self.out_mat > 1] = 1

        return {
            'loss_total': 0.0,
            'loss_current': 0.0,
            'loss_replay': 0.0,
            'pred': 0.0,
            'pred_re': 0.0,
            'distil_re': 0.0,
            'accuracy': accuracy
        }
