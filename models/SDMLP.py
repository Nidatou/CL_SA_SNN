import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

from models.cl.continual_learner import CLBase
from models.fc.nets import SDM_Base, Three_Layer_MLP


class SDM(CLBase):
    def __init__(
        self, input_size, nneu, output_size, device=None,
        # -> Top-K parameter
        k_mask=False, k_approach="LINEAR_DACAY", k_min=10, k_max=None,
        k_trans_ep=10000, gaba_switch_num=5000000, use_topK=True,
        # -> parameter_regularisation
        norm_ad=True, norm_val=True, dale=True,
    ):
        super(SDM, self).__init__()
        assert k_approach != "LINEAR_DECAY"
        self.flatten = nn.Flatten()
        self.use_topK = use_topK
        if use_topK:
            self.sdm_nets = SDM_Base(
                input_size=input_size, nneu=nneu, output_size=output_size, device=device,
                k_mask=k_mask, k_approach=k_approach, k_min=k_min, k_max=k_max,
                k_trans_ep=k_trans_ep, gaba_switch_num=gaba_switch_num,
                norm_ad=norm_ad, norm_val=norm_val, dale=dale,
            )
        else:
            self.sdm_nets = Three_Layer_MLP(
                input_size=input_size, nneu=nneu, output_size=output_size, device=device,
                norm_ad=norm_ad, norm_val=norm_val, dale=dale,
            )

        self.k_param = np.maximum(k_trans_ep, gaba_switch_num)
        self.cur_ep = 0

    @property
    def name(self):
        return self.sdm_nets.name

    @property
    def classifier_weight(self):
        if self.use_topK:
            return self.sdm_nets.purkinje_layer.weight.data.detach().cpu().numpy()
        else:
            return self.sdm_nets.output_layer.weight.data.detach().cpu().numpy()

    def enforce_weights_regulation(self):
        self.sdm_nets.enforce_weight_regularization()

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.flatten(x)
        x = self.sdm_nets(x, cur_ep=self.cur_ep, training=True)

        return x

    def classify(self, x: torch.Tensor, return_spk=False, **kwargs):
        x = self.flatten(x)
        output = self.sdm_nets(x, cur_ep=self.k_param, training=False)

        if return_spk:
            hid_opt = self.sdm_nets.mid_forward(x, cur_ep=self.k_param)
            return output, (hid_opt > 0).detach()

        return output

    def check_neuron_activation(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.flatten(x)
            hid_opt = self.sdm_nets.mid_forward(x, cur_ep=self.k_param)

        return (hid_opt > 0).detach()

    def train_a_batch(
        self, optimizer: torch.optim.Optimizer, x, y=None, x_=None, y_=None, scores_=None, rnt=0.5,
        active_classes=None, task=1, replay_not_hidden=False, freeze_convE=False, **kwargs
    ):
        self.train()

        optimizer.zero_grad()
        if x is not None:
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            y_hat = self(x)
            # temporally discard the active classes tag for test
            # if active_classes is not None:
            #     # for Task-IL and class-IL, model need to remove several output
            #     # and task-IL only keep the classes for current task
            #     class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            #     y_hat = y_hat[:, class_entries]
            if y is not None and len(y.size()) == 0:
                y = y.expand(1)
            predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='none')
            predL = None if y is None else torch.mean(predL, dim=0)

            # Weigh losses
            loss_cur = predL
            # Calculate training-accuracy
            accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

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
        # Take optimization-step
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        self.cur_ep += 1
        # -> possibly make the weight to be positive and on the surface of hyper-sphere
        self.enforce_weights_regulation()

        return {
            'loss_total': loss_total,
            'loss_current': loss_cur,
            'loss_replay': 0.0,
            'pred': predL,
            'pred_re': 0.0,
            'distil_re': 0.0,
            'accuracy': accuracy
        }




