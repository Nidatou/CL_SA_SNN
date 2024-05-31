from torch import nn
import torch
import random
import numpy as np

from models.cl.continual_learner import CLBase


class FlyModel(CLBase):
    def __init__(self, n_pn, n_kc, n_out, kc_prop=0.1, top_k=32, lr=1e-2, device=None):
        super(FlyModel, self).__init__()
        self.n_pn = n_pn
        self.n_kc = n_kc
        self.classes = n_out
        self.label = "Flymodel"
        self.pn_select = int(np.floor(n_pn * kc_prop))  # the number of pns summed up by every kc cell
        self.lr = lr

        self.kc_mat = torch.zeros((n_kc, n_pn)).to(device)
        self.out_mat = torch.zeros(n_out, n_kc).to(device)
        self.top_k = top_k
        for i in range(n_kc):
            self.kc_mat[i, random.sample(list(range(n_pn)), self.pn_select)] = 1
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            KCs = self.Generate_KCs(x)
            out = torch.matmul(KCs, self.out_mat.T)

        return out

    def classify(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            KCs = self.Generate_KCs(x)
            out = torch.matmul(KCs, self.out_mat.T)

        return out

    def Generate_KCs(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.flatten(x)
            KCs = torch.matmul(x, self.kc_mat.T)
            for i in range(KCs.shape[0]):
                sorted_kc, _ = torch.sort(KCs[i], descending=True)
                threshold = sorted_kc[self.top_k]
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
        return f"{self.label}_pn-{self.n_pn}_kc-{self.n_kc}-{self.pn_select}"

    def train_a_batch(
        self, optimizer: torch.optim.Optimizer, x, y=None, x_=None, y_=None, scores_=None, rnt=0.5,
        active_classes=None, task=1, replay_not_hidden=False, freeze_convE=False, scenario='class', **kwargs
    ):
        # 然而在这里optimser完全没有卵用...
        assert y is not None
        KCs = self.Generate_KCs(x)
        y_hat = torch.matmul(KCs, self.out_mat.T)
        accuracy = (y == y_hat.max(1)[1]).sum().item() / y.shape[0]

        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
        else:
            class_entries = list(range(self.classes))
        for i in class_entries:
            sum_kc = torch.sum(KCs[y+class_entries[0] == i], dim=0)
            self.out_mat[i] += self.lr * sum_kc
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
