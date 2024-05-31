from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import random

from models.cl.continual_learner import CLBase
from models.fc.kWTA_snn import kWTA_Spike, kWTA_Mem, kWTA_Spk_tr, Spike_Filter
from models.fc.nets import MLP
from models.fc.layers import fc_layer
from models.utils import utils
from models.utils.loss_func import TET_loss

from spikingjelly.activation_based import surrogate, neuron, layer, functional, encoding


class Gate_snn(CLBase):
    def __init__(
        self, input_size, hid_size, output_size, T=16, device=None,
        surr_func=surrogate.Sigmoid(), neu='if',
        # -> custom parameter for wta-k
        k_min=10, use_mask=True, step_mask=False, single_WTA=False, detach_nonspike=False, inhibit_intense=1.,
        # -> parameters for trace mask
        lateral_inh=True, self_inh=False, tr_decay=10., tr_refresh=True, one_size_tr=False, hard_mask=True,
        # -> parameters for adaptive threshold
        adaptive_thresh=False, adap_param=1e+5, adap_approach="linear", thresh_max=2.0, thresh_min=1.0,
        # -> parameters for k selection
        last_inhibit=False, curr_excite=False, tune_approach="sigmoid", inhibit_p=1e-4, excite_p=1e-5,
        tune_param=1.0, adapt_tune=True, k_param=200, k_approach="FLAT", k_max=None,
        # ->other parameters
        dale=True, norm_input=True, norm_fir=True, freeze_fir=False, final_neu="mem",
        loss_type="ce", loss_lamb=1e-3, loss_means=1.,  # -> setting for loss function
        spk_ipt=False, filter_prop=0.1, norm_const=1.,
    ):
        assert device is not None
        assert final_neu in ['mem', 'normal']
        super(Gate_snn, self).__init__()
        self.flatten = layer.Flatten()
        self.use_mask = use_mask
        self.freeze_fir = freeze_fir

        if self.use_mask:
            if not step_mask:
                self.topk = kWTA_Spike(
                    neu_size=hid_size, device=device, neuron_type=neu, surr_func=surr_func,
                    # -> parameters for threshold
                    adaptive_threshold=adaptive_thresh, adap_param=adap_param, thresh_max=thresh_max,
                    thresh_min=thresh_min, adap_approach=adap_approach,
                    # -> parameters for k_selection
                    last_inhibit=last_inhibit, curr_excite=curr_excite, tune_approach=tune_approach,
                    inhibit_p=inhibit_p, excite_p=excite_p, tune_param=tune_param, adapt_tune=adapt_tune,
                    # -> parameter for k-choose
                    k_min=k_min, k_param=k_param, k_approach=k_approach, k_max=k_max,
                    # -> other parameters
                    detach_nonspike=detach_nonspike,
                )
            else:
                self.topk = kWTA_Spk_tr(
                    neu_size=hid_size, device=device, neu_type=neu, surr_func=surr_func,
                    # -> parameter for k-WTA
                    k_min=k_min, inhibit_intense=inhibit_intense, use_inhibit=lateral_inh, inhibit_self=self_inh,
                    hard_mask=hard_mask, k_max=k_max, k_approach=k_approach, k_param=k_param,
                    # -> parameter for trace
                    tr_refresh=tr_refresh, tr_decay=tr_decay, one_size_tr=one_size_tr,
                    # -> parameters for adaptive threshold
                    adaptive_threshold=adaptive_thresh, adap_approach=adap_approach, adap_param=adap_param,
                    thresh_min=thresh_min, thresh_max=thresh_max
                )
        else:
            self.topk = kWTA_Mem(
                surr_func=surr_func, neu_size=hid_size, neu_type=neu, device=device,
                single_WTA=single_WTA, k_min=k_min, detach_nonspike=detach_nonspike,
                adaptive_threshold=adaptive_thresh, adap_param=adap_param,
                inhibit_intense=inhibit_intense, thresh_max=thresh_max, thresh_min=thresh_min,
            )
        if final_neu == "mem":
            self.final_neu = utils.NonSpikingIFNode
        else:
            self.final_neu = utils.Identity
        self.layer1 = layer.Linear(input_size, hid_size, bias=False)
        if freeze_fir:
            self.layer1.requires_grad_(False)
            selected = int(np.floor(0.1 * input_size))
            self.layer1.weight.data.zero_()
            for i in range(hid_size):
                self.layer1.weight.data[i, random.sample(list(range(input_size)), selected)] = T / selected

        self.layer2 = layer.Linear(hid_size, output_size, bias=False)
        self.out_neu = self.final_neu()

        self.dale = dale
        self.norm_input = norm_input
        self.norm_fir = norm_fir
        self.cur_ep = 0
        self.T = T
        self.task_spike_record = []

        assert loss_type in ['ce', 'tet']
        self.loss_type = loss_type
        self.loss_lamb = loss_lamb
        self.loss_means = loss_means
        if self.loss_type == 'tet':
            self.logits = None

        # 也许可能会用到的参数
        self.spk_ipt = spk_ipt
        self.norm_const = 1.
        if spk_ipt and filter_prop != 1.:
            self.filter = Spike_Filter(filter_prop=filter_prop)
            self.norm_const = np.sqrt(norm_const)
            print("filter defined")

        functional.set_step_mode(self, 'm')

        ipt_label = f"_poi-{filter_prop}_norm-{norm_const}" if self.spk_ipt else ""
        self.label = f"WTA_k{input_size}x{hid_size}x{output_size}_{'mem' if final_neu=='mem' else 'norm'}{ipt_label}"

    @property
    def name(self):
        return f"{self.label}-{self.topk.name}"

    def enforce_weights_regulation(self):
        # self.layer2.weight.data.clamp_(0)
        if self.dale:
            with torch.no_grad():
                # self.layer1.weight.data.clamp_(0)
                # self.layer2.weight.data.clamp_(0)
                for p in list(self.parameters()):
                    if p.requires_grad:
                        p.data.clamp_(0)
        # if self.norm_fir and not self.freeze_fir:
        self.layer1.weight.data /= (torch.norm(self.layer1.weight.data, dim=1, keepdim=True) * self.norm_const)

    def forward(self, x: torch.Tensor, training=True, **kwargs):
        if len(x.shape) == 4 or len(x.shape) == 2:
            x = x.repeat(self.T, *(len(x.shape) * [1]))
        x = self.flatten(x)
        if self.norm_input:
            if hasattr(self, 'filter'):
                x = utils.encoder()(x)
                x = self.filter(x)
            else:
                x = x / torch.norm(x, dim=-1, keepdim=True)
                if self.spk_ipt:
                    x = utils.encoder()(x)
            # x = self.encoder(x)  # -> 这个是实在没有办法了这么做肯定是没什么问题的，但是这个方法有点过于僵硬了
        hid_feature = self.layer1(x)
        hid_spike = self.topk(hid_feature, curr_ep=self.cur_ep, training=training)

        output = self.layer2(hid_spike)
        output = self.out_neu(output)
        if self.loss_type == 'tet' and kwargs.get('training', True):
            self.logits = output
        out = output[-1] if self.final_neu == utils.NonSpikingIFNode else torch.mean(output, dim=0)
        return out

    def classify(self, x: torch.Tensor, return_spk=False, **kwargs):
        with torch.no_grad():
            if len(x.shape) == 4 or len(x.shape) == 2:
                x = x.repeat(self.T, *(len(x.shape) * [1]))
            x = self.flatten(x)
            if self.norm_input:
                if hasattr(self, 'filter'):
                    x = utils.encoder()(x)
                    x = self.filter(x)
                else:
                    x = x / torch.norm(x, dim=-1, keepdim=True)
                    if self.spk_ipt:
                        x = utils.encoder()(x)
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
            if len(x.shape) == 4 or len(x.shape) == 2:
                x = x.repeat(self.T, *(len(x.shape) * [1]))
            x = self.flatten(x)
            if self.norm_input:
                if hasattr(self, 'filter'):
                    x = utils.encoder()(x)
                    x = self.filter(x)
                else:
                    x = x / torch.norm(x, dim=-1, keepdim=True)
                    if self.spk_ipt:
                        x = utils.encoder()(x)
            hid_feature = self.layer1(x)
            hid_spike = self.topk(hid_feature, curr_ep=self.cur_ep, training=False)
            functional.reset_net(self)

        return torch.sum(hid_spike, dim=0).detach()

    @property
    def classifier_weight(self):
        return self.layer2.weight.data.detach().cpu().numpy()

    @property
    def first_layer_weight(self):
        return self.layer1.weight.data.detach().cpu().numpy()

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
            # temporally discard the active classes tag for test
            # if active_classes is not None:
            #     # for Task-IL and class-IL, model need to remove several output
            #     # and task-IL only keep the classes for current task
            #     class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            #     y_hat = y_hat[:, class_entries]
            #     if self.loss_type == 'tet':
            #         self.logits = self.logits[..., class_entries]
            if y is not None and len(y.size()) == 0:
                y = y.expand(1)
            if self.loss_type == 'ce':
                predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='none')
                predL = None if y is None else torch.mean(predL, dim=0)
            elif self.loss_type == 'tet':
                predL = None if y is None else TET_loss(outputs=self.logits, labels=y, means=self.loss_means, lamb=self.loss_lamb)
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
            # -> for some conditions we need to deal with the top-k layers
            self.task_spike_record.append(self.topk.neu_act_cnts)
            self.topk.end_one_task()
            if len(self.task_spike_record) == 5:
                final_res = torch.cat(self.task_spike_record, dim=0).cpu().numpy()
                np.save('./store/debug_info/spike_act.npy', final_res)
                print(f"activate record saved")

        return {
            'loss_total': loss_total,
            'loss_current': loss_cur,
            'loss_replay': 0.0,
            'pred': predL,
            'pred_re': 0.0,
            'distil_re': 0.0,
            'accuracy': accuracy
        }

