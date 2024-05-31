import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class kWTA_ann(nn.Module):
    def __init__(
        self, neu_size, device=None, use_mask=False,
        # -> parameter for the k's choice
        k_min=10, k_param=1000, k_approach="LINEAR_DECAY"
    ):
        super(kWTA_ann, self).__init__()
        self.neu_size = neu_size
        self.nl = nn.ReLU()

        # some parameters for k_approach
        self.k_approach = k_approach
        assert self.k_approach in ['LINEAR_DECAY', 'FLAT', 'GABA']
        self.k_min = k_min
        self.k_max = self.neu_size
        self.k_param = k_param
        self.use_mask = use_mask

        self.last_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)
        self.neu_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)

        #################################
        # generate a name for the model #
        #################################
        special_label = "sub" if not self.use_mask else 'mask'
        k_label = "-{approach}{parameters}".format(
            approach="F_" if self.k_approach == "FLAT" else ("LD_" if self.k_approach=="LINEAR_DECAY" else "GB_"),
            parameters=f"{self.k_min}" if self.k_approach == "FLAT"
            else f"p{int(self.k_param)}"
        )
        self.label = f"-{special_label}{k_label}"

    @property
    def name(self):
        return self.label

    def get_curr_k(self, curr_ep=None):

        if curr_ep is None:
            return self.k_min
        if self.k_approach == "LINEAR_DECAY":
            k_max = self.k_max
            linear_coef = (-(k_max - self.k_min) / self.k_param)
            k = np.minimum(
                k_max,
                np.maximum(
                    k_max + curr_ep * linear_coef, self.k_min,
                ),
            )
            k = int(k)
        else:
            k = self.k_min

        if self.k_approach == "GABA":
            k = k + 1

        return k

    def forward(self, x: torch.Tensor, curr_ep=None, training=True):
        x = self.nl(x)
        curr_k = self.get_curr_k(curr_ep=curr_ep)

        vals, inds = torch.topk(x, np.minimum(curr_k, self.k_max), dim=-1, sorted=False)
        inhib_sig, _ = torch.min(vals.detach(), dim=-1, keepdim=True)

        if self.k_approach == "GABA":
            linear_coef = 2 / self.k_param
            gaba_response = torch.minimum(
                torch.ones_like(self.neu_act_cnts),
                torch.maximum(
                    -1 + (linear_coef * (self.neu_act_cnts + self.last_act_cnts)),
                    torch.ones_like(self.neu_act_cnts) * -1,
                ),
            ).type_as(x)
        else:
            gaba_response = 1

        if self.use_mask:
            top_k_mask = torch.zeros_like(x)
            top_k_mask = top_k_mask.scatter(-1, inds, 1)
            x = x * top_k_mask
        else:
            x = self.nl(x - (gaba_response * inhib_sig))
            self.neu_act_cnts += torch.sum(x>0, dim=0, keepdim=True)

        return x

    # -> accumulate the spike sum once a task is ended
    def end_one_task(self):
        self.last_act_cnts += self.neu_act_cnts
        self.neu_act_cnts = torch.zeros_like(self.neu_act_cnts)
        if self.adapt_tune:
            self.excite_p = self.excite_p * (self.task_num + 1) / self.task_num
            self.task_num += 1
