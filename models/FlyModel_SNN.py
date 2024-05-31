from torch import nn
import torch
import random
import numpy as np

from models.cl.continual_learner import CLBase
from utils.utils import minmax_val
from spikingjelly.activation_based import neuron, layer, surrogate, functional


class FlyModel_SNN(CLBase):
    def __init__(
        self, image_size, image_channels, output_size, device=None,
        n_kc=1000, kc_response=32, k_num=32, fly_lr=5e-3, min_max=True,
        norm_const=1., T=16, tr_decay=10.,
    ):
        assert device is not None
        super(FlyModel_SNN, self).__init__()

        self.conv_out_units = image_channels * image_size ** 2
        self.n_kc = n_kc
        self.classes = output_size
        self.label = "SNN_fly"
        self.kc_response = kc_response
        self.fly_lr = fly_lr
        self.min_max = min_max
        self.norm_const = np.sqrt(norm_const)

        self.kc_mat = torch.zeros((n_kc, self.conv_out_units)).to(device)
        self.kc_tr = None
        self.hid_spk = []
        self.out_mat = torch.zeros(self.classes, n_kc).to(device)
        self.k_num = k_num
        self.kc_neu = neuron.IFNode()
        self.tr_decay = tr_decay
        for i in range(n_kc):
            self.kc_mat[i, random.sample(list(range(self.conv_out_units)), self.kc_response)] = 1
        self.flatten = layer.Flatten()

        self.kc_mat.data /= (torch.norm(self.kc_mat.data, dim=1, keepdim=True) * self.norm_const)
        self.T = T
        functional.set_step_mode(self, 'm')
        functional.reset_net(self)

    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            if len(x.shape) == 2 or len(x.shape) == 4:
                x = x.repeat(self.T, *(len(x.shape) * [1]))
            x_seq = self.flatten(x)
            x_seq = x_seq / torch.norm(x, dim=-1, keepdim=True)
            self.hid_spk = []
            if self.kc_tr is None:
                self.kc_tr = torch.zeros(self.n_kc).to(self._device())
            for t in range(self.T):
                mem_seq = torch.matmul(x_seq[t], self.kc_mat.T)
                spk_seq = self.kc_neu(mem_seq)
                # generate the Top-K mask
                append_tr = spk_seq.detach()
                if self.tr_decay != 0:
                    self.kc_tr = self.kc_tr - self.kc_tr / self.tr_decay + append_tr
                else:
                    self.kc_tr = self.kc_tr + append_tr
                vals, inds = torch.topk(self.kc_tr, self.k_num + 1, dim=-1, sorted=False)
                thresh, _ = torch.min(vals.detach(), dim=-1, keepdim=True)
                top_k_mask = (self.kc_tr > thresh).float()

                spk_out = top_k_mask * spk_seq
                self.hid_spk.append(spk_out)
            self.hid_spk = torch.stack(self.hid_spk)
            y_out = torch.matmul(self.hid_spk, self.out_mat.T)

        self.kc_tr = None
        functional.reset_net(self)
        return torch.mean(y_out, dim=0)

    def classify(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            if len(x.shape) == 2 or len(x.shape) == 4:
                x = x.repeat(self.T, *(len(x.shape) * [1]))
            x_seq = self.flatten(x)
            x_seq = x_seq / torch.norm(x, dim=-1, keepdim=True)
            hid_spk = []
            if self.kc_tr is None:
                self.kc_tr = torch.zeros(self.n_kc).to(self._device())
            for t in range(self.T):
                mem_seq = torch.matmul(x_seq[t], self.kc_mat.T)
                spk_seq = self.kc_neu(mem_seq)

                append_tr = spk_seq.detach()
                if self.tr_decay != 0:
                    self.kc_tr = self.kc_tr - self.kc_tr / self.tr_decay + append_tr
                else:
                    self.kc_tr = self.kc_tr + append_tr
                vals, inds = torch.topk(self.kc_tr, self.k_num + 1, dim=-1, sorted=False)
                thresh, _ = torch.min(vals.detach(), dim=-1, keepdim=True)
                top_k_mask = (self.kc_tr > thresh).float()

                spk_out = top_k_mask * spk_seq
                hid_spk.append(spk_out)
            hid_spk = torch.stack(hid_spk)
            y_out = torch.matmul(hid_spk, self.out_mat.T)

        self.kc_tr = None
        functional.reset_net(self)
        return torch.mean(y_out, dim=0)

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
        assert y is not None
        x = self.flatten(x)
        if self.min_max:
            x = minmax_val(x, dim=-1)
        y_hat = self(x)
        accuracy = (y == y_hat.max(1)[1]).sum().item() / y.shape[0]

        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
        else:
            class_entries = list(range(self.classes))

        for i in class_entries:
            total_spk = torch.sum(self.hid_spk, dim=0)
            sum_kc = torch.sum(total_spk[y+class_entries[0] == i], dim=0)
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



