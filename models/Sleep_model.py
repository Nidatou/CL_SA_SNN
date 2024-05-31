import tqdm

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import random

from models.cl.continual_learner import CLBase
from models.fc.kWTA_snn import kWTA_Spike, kWTA_Spk_tr, kWTA_Spk_tr_inh, Spike_Filter
from models.utils import utils
from models.utils.loss_func import TET_loss, softmax_cross_entropy

from spikingjelly.activation_based import surrogate, neuron, layer, functional, encoding


class Sleep_snn(CLBase):
    def __init__(
        self, input_size, output_size, T=16, device=None, surr_func=surrogate.Sigmoid(), neu='if',
        # -> custom parameters for wta_K
        k_min=10, step_mask=False, tr_decay=10., hard_mask=False,
        # -> parameters for adaptive threshold
        adaptive_thresh=False, adap_param=1e+5, adap_approach="linear", thresh_max=2.0, thresh_min=1.0,
        # -> parameters for k selection
        last_inhibit=False, curr_excite=False, inhibit_p=1e-6, excite_p=1e-4, tune_param=1.0, adapt_tune=True,
        # fc_part
        h_dim=1000, final_neu='neu',
        # sleep unsupervised learning
        inc_lr=1e-4, dec_lr=1e-4, sleep_times=1024, sleep_batch=128, need_sleep=True, topk_sleep=True,
        # other setting
        loss_type='ce', loss_lamb=1e-3, loss_means=1., dale=True, norm_weight=True, norm_input=True, norm_const=1.,
        filter_prop=0.1, spk_ipt=True,
    ):
        assert device is not None
        assert final_neu in ['mem', 'normal', 'neu']
        assert neu in ['if', 'lif']
        super(Sleep_snn, self).__init__()

        self.surr_func = surr_func
        self.flatten = layer.Flatten()
        self.fc_neu = neuron.IFNode if neu == 'if' else neuron.LIFNode
        self.dale = dale
        self.norm_weight = norm_weight
        self.norm_input = norm_input

        if not step_mask:
            self.topk = kWTA_Spike(
                neu_size=h_dim, device=device, neuron_type=neu, surr_func=surr_func,
                # -> parameters for threshold
                adaptive_threshold=adaptive_thresh, adap_param=adap_param, thresh_max=thresh_max,
                thresh_min=thresh_min, adap_approach=adap_approach,
                # -> parameters for k_selection
                last_inhibit=last_inhibit, curr_excite=curr_excite, inhibit_p=inhibit_p, excite_p=excite_p,
                tune_param=tune_param, adapt_tune=adapt_tune,
                # -> parameter for k-choose
                k_min=k_min,
            )
        else:
            self.topk = kWTA_Spk_tr_inh(
                neu_size=h_dim, surr_func=surr_func, neu_type=neu, device=device,
                # -> parameter for k-WTA
                k_min=k_min, hard_mask=hard_mask, tr_decay=tr_decay,
                # -> parameter for specific excite and inhibit
                last_inhibit=last_inhibit, curr_excite=curr_excite, inhibit_p=inhibit_p, excite_p=excite_p,
                tune_param=tune_param, adapt_tune=adapt_tune,
                # -> parameters for adaptive threshold
                adaptive_threshold=adaptive_thresh, adap_approach=adap_approach, adap_param=adap_param,
                thresh_min=thresh_min, thresh_max=thresh_max,
            )

        self.input_size = input_size
        self.h_dim = h_dim

        self.layer1 = layer.Linear(input_size, h_dim, bias=False)
        self.layer2 = layer.Linear(h_dim, output_size, bias=False)
        if final_neu == 'mem':
            self.final_neu = utils.NonSpikingIFNode
        elif final_neu == 'normal':
            self.final_neu = utils.Identity
        else:
            self.final_neu = self.fc_neu
        assert final_neu == 'normal'
        self.out_neu = self.final_neu(surrogate_function=surr_func)

        self.spk_ipt = spk_ipt
        self.filter_prop = filter_prop
        self.norm_const = np.sqrt(norm_const)
        if spk_ipt and filter_prop < 1.:
            self.filter = Spike_Filter(filter_prop=filter_prop)
            print('filter defined')

        self.topk_sleep = topk_sleep
        self.inc_lr = inc_lr
        self.dec_lr = dec_lr
        self.need_sleep = need_sleep
        self.sleep_times = sleep_times
        self.sleep_batch = 1
        self.mask_size = 100
        self.T = T
        self.noise_plat = None
        self.remem_num = 0

        self.loss_type = loss_type
        self.loss_lamb = loss_lamb
        self.loss_means = loss_means
        if self.loss_type == 'tet':
            self.logits = None

        functional.set_step_mode(self, 'm')

        ipt_label = f"_poi-{filter_prop}_norm-{norm_const}" if self.spk_ipt else ""
        self.model_label = f"{input_size}x{h_dim}x{output_size}"
        self.kWTA_label = self.topk.name
        self.spe_label = f"-sleep_b{self.sleep_batch}_t{self.sleep_times}{'-nt' if not self.topk_sleep else ''}"
        self.label_name = "Sleep_net"
        self.label = f"{self.label_name}-{self.kWTA_label}-{self.model_label}" \
                     f"{self.spe_label if self.need_sleep else ''}{ipt_label}"

    @property
    def name(self):
        return self.label

    def enforce_weights_regulation(self):
        if self.dale:
            with torch.no_grad():
                for p in list(self.parameters()):
                    if p.requires_grad:
                        p.data.clamp_(0)
        if self.norm_weight:
            self.layer1.weight.data /= (torch.norm(self.layer1.weight.data, dim=1, keepdim=True) * self.norm_const)

    def forward(self, x: torch.Tensor, training=True, **kwargs):
        #######################################
        # append the noise tensor
        if training:
            added_noise = torch.mean(x, dim=0)  # -> change to the one-dim noise
            added_noise = torch.flatten(added_noise)
            assert len(added_noise.shape) == 1
            if self.noise_plat is None:
                self.noise_plat = added_noise.detach()
            else:
                self.noise_plat += added_noise.detach()
            self.remem_num += 1
        #######################################
        if len(x.shape) == 4 or len(x.shape) == 2:
            x = x.repeat(self.T, *(len(x.shape) * [1]))
        x = self.flatten(x)
        if self.norm_input:
            x = x / torch.norm(x, dim=-1, keepdim=True)
        if self.spk_ipt:
            x = utils.encoder()(x)
            if self.filter_prop < 1.:
                x = self.filter(x)

        hid_feature = self.layer1(x)
        hid_spk = self.topk(hid_feature, training=training)
        output = self.out_neu(self.layer2(hid_spk))
        if self.loss_type == 'tet' and training:
            self.logits = output
        out = torch.mean(output, dim=0) if self.final_neu==utils.Identity else output[-1]
        return out

    def classify(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            if len(x.shape) == 4 or len(x.shape) == 2:
                x = x.repeat(self.T, *(len(x.shape) * [1]))
            x = self.flatten(x)
            if self.norm_input:
                x = x / torch.norm(x, dim=-1, keepdim=True)
            if self.spk_ipt:
                x = utils.encoder()(x)
                if self.filter_prop < 1.:
                    x = self.filter(x)

            hid_feature = self.layer1(x)
            hid_spk = self.topk(hid_feature, training=False)
            output = self.out_neu(self.layer2(hid_spk))
            out = output[-1] if self.final_neu == utils.NonSpikingIFNode else torch.mean(output, dim=0)
            functional.reset_net(self)
        return out

    def one_step_sleep(self, input_noise, training=False):
        with torch.no_grad():
            one_normal_neu = neuron.IFNode(surrogate_function=self.surr_func, step_mode='m')
            if self.norm_input:
                input_noise = input_noise / torch.norm(input_noise, dim=-1, keepdim=True)
            if self.spk_ipt:
                poi_input = utils.encoder()(input_noise)
                if self.filter_prop < 1.:
                    poi_input = self.filter(poi_input)

            hid_feature = self.layer1(poi_input)
            if self.topk_sleep:
                hid_spk = self.topk(hid_feature, training=training)
            else:
                hid_spk = one_normal_neu(hid_feature)

            poi_input = poi_input.view(-1, poi_input.shape[-1])
            hid_spk = hid_spk.view(-1, hid_spk.shape[-1])
            inc_layer1 = torch.sum(hid_spk.unsqueeze(2) * poi_input.unsqueeze(1), dim=0)
            dec_layer1 = torch.sum(hid_spk.unsqueeze(2) * (1 - poi_input).unsqueeze(1), dim=0)

            self.layer1.weight.data += self.inc_lr * inc_layer1 - self.dec_lr * dec_layer1
            functional.reset_net(self)

    def sleep_stage(self):
        if not self.need_sleep:
            print("need not to sleep")
            return
        print("Begin to sleep")
        assert self.remem_num > 0
        noise_aplat = self.noise_plat / self.remem_num
        noise_aplat = noise_aplat.repeat(self.T, self.sleep_batch, 1)
        for t in range(self.sleep_times):
            # random_mask = torch.zeros(self.input_size, ).to(self._device())
            # x_pos = np.random.randint(0, self.input_size - self.mask_size)
            # random_mask[x_pos: x_pos + self.mask_size] = 1
            # masked_noise = noise_aplat * random_mask
            masked_noise = noise_aplat
            self.one_step_sleep(masked_noise, training=False)
            # poi_input = utils.encoder()(masked_noise)
            # hid_spk = self.neu1(self.layer1(poi_input))
            # opt_spk = self.out_neu(self.layer2(hid_spk))
            #
            # poi_input = poi_input.view(-1, poi_input.shape[-1])
            # hid_spk = hid_spk.view(-1, hid_spk.shape[-1])
            # opt_spk = opt_spk.view(-1, opt_spk.shape[-1])
            #
            # inc_layer1 = torch.sum(hid_spk.unsqueeze(2) * poi_input.unsqueeze(1), dim=0)
            # inc_layer2 = torch.sum(opt_spk.unsqueeze(2) * hid_spk.unsqueeze(1), dim=0)
            # dec_layer1 = torch.sum(hid_spk.unsqueeze(2) * (1 - poi_input).unsqueeze(1), dim=0)
            # dec_layer2 = torch.sum(opt_spk.unsqueeze(2) * (1 - hid_spk).unsqueeze(1), dim=0)
            #
            # self.layer1.weight.data += self.inc_lr * inc_layer1 - self.dec_lr * dec_layer1
            # self.layer2.weight.data += self.inc_lr * inc_layer2 - self.dec_lr * dec_layer2
            self.enforce_weights_regulation()

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
                # predL = softmax_cross_entropy(output=y_hat, y=y, beta=self.beta, reduction='mean')
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
        # -> possibly make the weight to be positive and on the surface of hyper-sphere
        self.enforce_weights_regulation()

        if one_task_ended:
            # -> passing a sleeping period
            self.topk.end_one_task()
            self.sleep_stage()

        return {
            'loss_total': loss_total,
            'loss_current': loss_cur,
            'loss_replay': 0.0,
            'pred': predL,
            'pred_re': 0.0,
            'distil_re': 0.0,
            'accuracy': accuracy
        }

