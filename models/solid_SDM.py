from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import random

from models.cl.continual_learner import CLBase
from models.fc.kWTA_snn import kWTA_Spike, kWTA_Spk_tr
from models.utils import utils
from models.utils.loss_func import TET_loss

from spikingjelly.activation_based import surrogate, neuron, layer, functional, encoding


class Solid_Gate(CLBase):
    def __init__(
        self, image_size, image_channels, output_size, T=16, device=None, surr_func=surrogate.Sigmoid(),
        ######################################################
        # fc_kWTA_layers-parts
        h_dim=1000, fc_neu='if', k_min=10, step_mask=False,
        # -> parameters for trace mask
        tr_decay=10., hard_mask=True,
        # -> parameters for adaptive threshold
        adaptive_thresh=False, adap_param=1e+5, adap_approach="linear", thresh_max=2.0, thresh_min=1.0,
        # -> parameters for k selection
        # -> other parameters
        dale=True, norm_input=True, norm_fir=True, final_neu="mem",
        loss_type='ce', loss_lamb=1e-3, loss_means=1., norm_const=1., p=1
    ):
        assert device is not None
        assert final_neu in ['mem', 'normal']
        super(Solid_Gate, self).__init__()
        self.classes = output_size

        self.flatten = layer.Flatten()
        self.fc_input_size = image_size ** 2 * image_channels
        # FC Layers #############
        if step_mask:
            self.topk = kWTA_Spk_tr(
                neu_size=h_dim, device=device, neu_type=fc_neu, surr_func=surr_func,
                # -> parameter for k-WTA
                k_min=k_min, hard_mask=hard_mask, tr_decay=tr_decay,
                # -> parameters for adaptive threshold
                adaptive_threshold=adaptive_thresh, adap_approach=adap_approach, adap_param=adap_param,
                thresh_min=thresh_min, thresh_max=thresh_max,
            )
        else:
            self.topk = kWTA_Spike(
                neu_size=h_dim, device=device, neuron_type=fc_neu, surr_func=surr_func,
                # -> parameters for adaptive threshold
                adaptive_threshold=adaptive_thresh, adap_param=adap_param, thresh_max=thresh_max,
                thresh_min=thresh_min, adap_approach=adap_approach,
            )
        if final_neu == 'mem':
            self.final_neu = utils.NonSpikingIFNode
        else:
            self.final_neu = utils.Identity
        self.layer1 = layer.Linear(self.fc_input_size, h_dim, bias=False)
        self.layer2 = layer.Linear(h_dim, output_size, bias=False)
        self.out_neu = self.final_neu(surrogate_function=surr_func)

        # Other setting #############
        self.dale = dale
        self.norm_input = norm_input
        self.norm_fir = norm_fir
        self.cur_ep = 0
        self.T = T

        assert loss_type in ['ce', 'tet']
        self.loss_type = loss_type
        self.loss_lamb = loss_lamb
        self.loss_means = loss_means
        if self.loss_type == 'tet':
            self.logits = None  # -> use to store the logits to calculate the loss

        self.norm_const = norm_const
        self.norm_p = p
        functional.set_step_mode(self, 'm')

        ablation_label = f"{'_no-dale' if self.dale is False else ''}{'_no-norm' if self.norm_input is False and self.norm_fir is False else ''}"
        self.label = f"Solid_WTA({self.norm_p}-{self.norm_const}){self.fc_input_size}x{h_dim}x{output_size}_{'mem' if final_neu == 'mem' else 'norm'}{ablation_label}"

    @property
    def name(self):
        return f"{self.label}-{self.topk.name}"

    def enforce_weights_regulation(self):
        if self.dale:
            with torch.no_grad():
                for p in list(self.parameters()):
                    if p.requires_grad:
                        p.data.clamp_(0)
        if self.norm_fir:
            self.layer1.weight.data /= (torch.norm(self.layer1.weight.data, p=self.norm_p, dim=1, keepdim=True) * self.norm_const)

    def forward(self, x: torch.Tensor, **kwargs):
        if self.dale:
            x = nn.ReLU()(x)
        if len(x.shape) == 4 or len(x.shape) == 2:
            x = x.repeat(self.T, *(len(x.shape) * [1]))
        x = self.flatten(x)
        if self.norm_input:
            x = x / torch.norm(x, p=self.norm_p, dim=-1, keepdim=True)

        hid_feature = self.layer1(x)
        hid_spike = self.topk(hid_feature, curr_ep=self.cur_ep, training=kwargs.get('training', True))

        output = self.layer2(hid_spike)
        output = self.out_neu(output)
        if self.loss_type == 'tet' and kwargs.get('training', True):
            self.logits = output
        out = output[-1] if self.final_neu == utils.NonSpikingIFNode else torch.mean(output, dim=0)
        return out

    def classify(self, x: torch.Tensor, return_spk=False, **kwargs):
        with torch.no_grad():
            if self.dale:
                x = nn.ReLU()(x)
            if len(x.shape) == 4 or len(x.shape) == 2:
                x = x.repeat(self.T, *(len(x.shape) * [1]))
            x = self.flatten(x)
            if self.norm_input:
                x = x / torch.norm(x, p=self.norm_p, dim=-1, keepdim=True)

            hid_feature = self.layer1(x)
            hid_spike = self.topk(hid_feature, curr_ep=self.cur_ep, training=False)

            output = self.layer2(hid_spike)
            output = self.out_neu(output)
            out = output[-1] if self.final_neu == utils.NonSpikingIFNode else torch.mean(output, dim=0)
            functional.reset_net(self)

        if return_spk:
            return out, torch.sum(hid_spike, dim=0).detach()
        return out

    def check_neuron_activation(self, x: torch.Tensor):
        with torch.no_grad():
            if self.dale:
                x = nn.ReLU()(x)
            if len(x.shape) == 4 or len(x.shape) == 2:
                x = x.repeat(self.T, *(len(x.shape) * [1]))
            x = self.flatten(x)
            if self.norm_input:
                x = x / torch.norm(x, p=self.norm_p, dim=-1, keepdim=True)

            hid_feature = self.layer1(x)
            hid_spike = self.topk(hid_feature, curr_ep=self.cur_ep, training=False)
            functional.reset_net(self)

        return torch.sum(hid_spike, dim=0).detach()

    @property
    def classifier_weight(self):
        return self.layer2.weight.data.detach().cpu().numpy()

    def train_a_batch(
        self, optimizer: torch.optim.Optimizer, x, y=None, x_=None, y_=None, scores_=None, rnt=0.5,
        active_classes=None, task=1, replay_not_hidden=False, freeze_convE=False, scenario='class',
        one_task_ended=False, **kwargs
    ):
        self.train()
        optimizer.zero_grad()
        if x is not None:
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            if len(x.shape) == 4 or len(x.shape) == 2:
                batch_size = x.shape[0]
            else:
                batch_size = x.shape[1]
            y_hat = self(x)
            if y is not None and len(y.size()) == 0:
                y = y.expand(1)
            if self.loss_type == 'ce':
                predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='none')
                predL = None if y is None else torch.mean(predL, dim=0)
            elif self.loss_type == 'tet':
                assert self.logits is not None
                predL = None if y is None else TET_loss(outputs=self.logits, labels=y, means=self.loss_means, lamb=self.loss_lamb)
                self.logits = None
            else:
                raise NotImplementedError(f"loss type <{self.loss_type}> is not supported")

            loss_cur = predL
            accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item()/batch_size
            # 在采用了XdG情况下，反向传播需要在mask更换之前提前进行反向传播
            if (self.mask_dict is not None) and (x_ is not None):
                weighted_current_loss = rnt * loss_cur
                weighted_current_loss.backward()

        else:
            accuracy = predL = None

        # temporally do not consider
        if x_ is not None:
            pass

        loss_replay = None
        loss_total = None if x is None else loss_cur

        ########################################
        # loss in allocation (for regularization method) ######
        # Add SI-loss
        surrogate_loss = self.surrogate_loss()
        if self.si_c > 0:
            loss_total += self.si_c * surrogate_loss

        ewc_loss = self.ewc_loss()
        if self.ewc_lambda > 0:
            loss_total += self.ewc_lambda * ewc_loss

        mas_loss = self.mas_loss()
        if self.mas_lambda > 0:
            loss_total += self.mas_lambda * mas_loss

        if (self.mask_dict is None) or x_ is not None:
            loss_total.backward()

        functional.reset_net(self)
        ###########################################
        # Take optimization-step
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        optimizer.step()
        self.cur_ep += 1
        # -> possibly make the weight to be positive and on the surface of hyper-sphere
        self.enforce_weights_regulation()

        if one_task_ended:
            self.topk.end_one_task()

        return {
            'loss_total': loss_total,
            'loss_current': loss_cur,
            'loss_replay': 0.0,
            'pred': predL,
            'pred_re': 0.0,
            'distil_re': 0.0,
            'accuracy': accuracy
        }



