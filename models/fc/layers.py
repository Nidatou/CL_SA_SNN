import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np

from spikingjelly.activation_based import neuron, layer, surrogate, base

from models.utils import utils


class RecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, in_features: int, out_features: int, bias: bool = True,
                 step_mode='s', recur_val=1) -> None:
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'
        self.out_features = out_features
        self.rc = nn.Linear(in_features + out_features, out_features, bias)
        self.sub_module = sub_module
        self.recur_val = recur_val
        self.register_memory('y', None)

    def single_step_forward(self, x: torch.Tensor):
        if self.y is None:
            if x.ndim == 2:
                self.y = torch.ones([x.shape[0], self.sub_module_out_features]).to(x)
                self.y *= self.recur_val
            else:
                out_shape = [x.shape[0]]
                out_shape.extend(x.shape[1:-1])
                out_shape.append(self.sub_module_out_features)
                self.y = torch.ones(out_shape).to(x)
                self.y *= self.recur_val
        x = torch.cat((x, self.y), dim=-1)
        self.y = self.sub_module(self.rc(x))
        return self.y

    def extra_repr(self) -> str:
        return f', step_mode={self.step_mode}'


class Gated_Layer(nn.Module):
    def __init__(self, out_features, gate_buffer=False):
        super(Gated_Layer, self).__init__()
        buffer = torch.Tensor(out_features).uniform_(1, 1) if gate_buffer else None
        self.register_buffer("gate_buffer", buffer)

    def forward(self, x: torch.Tensor):
        if self.gate_buffer is None:
            excitability = 1
        else:
            excitability = self.gate_buffer
        return x * excitability


class fc_attention(nn.Module):
    def __init__(self, in_size, out_size, drop=0., neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid(),
                 forward_method='normal'):
        assert forward_method in ['normal', 'recurrent']
        super().__init__()
        self.drop = layer.Dropout(drop)
        if forward_method == 'normal':
            self.linear = nn.Sequential(
                layer.Linear(in_size, out_size, bias=False),
                neuron_type(surrogate_function=surr_func),
            )
        else:
            self.linear = RecurrentContainer(
                neuron_type(surrogate_function=surr_func, detach_reset=True), in_size, out_size, bias=False)
            self.recurrent_mask = torch.ones_like(self.linear.rc.weight.data)
            for i in range(out_size):
                self.recurrent_mask[i, in_size+i] = 0

    def forward(self, x: torch.Tensor, **kwargs):
        input = self.drop(x)
        output = self.linear(input)

        return output

    def cut_down_connection(self):
        if isinstance(self.linear, layer.LinearRecurrentContainer):
            with torch.no_grad():
                self.linear.rc.weight.data *= self.recurrent_mask


class fc_layer(nn.Module):
    '''Fully connected layer, with possibility of returning "pre-activations".

    Input:  [batch_size] x ... x [in_size] tensor
    Output: [batch_size] x ... x [out_size] tensor'''

    def __init__(self, in_size, out_size, drop=0., gate_buffer=True,
                 neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid()):
        super().__init__()
        self.dropout = layer.Dropout(drop)
        self.linear = layer.Linear(in_size, out_size, bias=False)
        self.final_neuron = neuron_type(surrogate_function=surr_func)
        self.gate_layer = Gated_Layer(out_features=out_size, gate_buffer=gate_buffer)

    def forward(self, x, **kwargs):
        input = self.dropout(x)
        pre_activ = self.linear(input)
        gated_pre_activ = pre_activ
        output = self.final_neuron(gated_pre_activ)
        output = self.gate_layer(output)
        return output

    def list_init_layers(self):
        return [self.linear]


# Fully connected layer outputting [mean] and [logvar] for each unit.
# input:[T, B, W]
class fc_layer_split(nn.Module):
    def __init__(self, in_size, out_size, gate_buffer=True, drop=0., neu_mean=utils.NonSpikingIFNode,
                 neu_logvar=utils.NonSpikingIFNode, surr_func=surrogate.Sigmoid()):
        super(fc_layer_split, self).__init__()
        self.mean = fc_layer(in_size, out_size, gate_buffer=gate_buffer, drop=drop, neuron_type=neu_mean, surr_func=surr_func)
        self.logvar = fc_layer(in_size, out_size, gate_buffer=gate_buffer, drop=drop, neuron_type=neu_logvar, surr_func=surr_func)

    def forward(self, x: torch.Tensor):
        return self.mean(x), self.logvar(x)

    def list_init_layers(self):
        lyr_list = []
        lyr_list += self.mean.list_init_layers()
        lyr_list += self.logvar.list_init_layers()
        return list


class fc_layer_fixed_gates(nn.Module):
    def __init__(self, in_size, out_size, gate_buffer=True, drop=0., neuron_type=neuron.IFNode,
                 surr_func=surrogate.Sigmoid(), gate_size=0, gating_prop=0.8, device="cuda:0"):
        super(fc_layer_fixed_gates, self).__init__()
        self.dropout = layer.Dropout(drop)
        self.linear = layer.Linear(in_size, out_size, bias=False)

        if gate_size > 0:
            self.gate_mask = torch.tensor(
                np.random.choice([0., 1.], size=(gate_size, out_size), p=[gating_prop, 1.-gating_prop]),
                dtype=torch.float, device=device
            )
        self.final_neuron = neuron_type(surrogate_function=surr_func)
        # self.gate_layer = Gated_Layer(out_features=out_size, gate_buffer=gate_buffer)

    def forward(self, x: torch.Tensor, gate_input=None):
        input = self.dropout(x)
        mem = self.linear(input)
        spike = self.final_neuron(mem)
        gate = torch.mm(gate_input, self.gate_mask) if hasattr(self, 'gate_mask') else None
        gated_spike = gate * spike if hasattr(self, 'gate_mask') else spike
        return gated_spike

    def list_init_layers(self):
        return [self.linear]


# A added WTA-k layer for SDM
class Top_K(nn.Module):
    def __init__(self, nneu, k_approch, k_min, k_max=None, k_trans_ep=None, gaba_switch_num=None, mask=False, device=None):
        super(Top_K, self).__init__()
        self.Relu = nn.ReLU()
        assert k_approch in ["LINEAR_DECAY", "GABA_SWITCH", "FLAT"]
        self.k_approch = k_approch
        self.k_max = nneu if k_max is None or k_max > nneu else k_max
        self.k_min = k_min
        self.k_trans_ep = k_trans_ep
        self.nneu = nneu
        self.use_mask = mask

        self.neuron_activation_counters = torch.zeros((1, nneu), requires_grad=False).to(device)
        if self.k_approch == "GABA_SWITCH":
            # When GABA, we need to count the activation times
            self.use_mask = False
            self.linear_coef_threshold = gaba_switch_num

    # <k_trans_ep> and <gaba_switch_num> are hyperparameters for Linear-decay and GABA switch
    # for "Linear_decay" method, we should calculate the K for TopK algorithm
    def get_curr_k(self, cur_ep):
        # -> consider the test stage
        if cur_ep is None:
            return self.k_min

        if "LINEAR_DECAY" in self.k_approch:
            k_max = self.k_max
            linear_coef = (-(k_max - self.k_min) / self.k_trans_ep)
            k = np.minimum(
                k_max,
                np.maximum(
                    k_max + cur_ep * linear_coef, self.k_min,
                ),
            )
            k = int(k + 0.5)
        else:
            k = self.k_min

        if "GABA_SWITCH" in self.k_approch:
            k = k + 1

        return k

    def forward(self, x: torch.Tensor, k_dim=1, cur_ep=None, training=True):
        assert cur_ep is not None
        x = self.Relu(x)
        curr_k = self.get_curr_k(cur_ep=cur_ep)  # -> calculate the number in WTA

        # calculate the threshold
        vals, inds = torch.topk(x, np.minimum(curr_k, self.k_max), dim=-1, sorted=False)
        inhib_sig, _ = torch.min(vals.detach(), dim=-1, keepdim=True)

        # <gaba_reponse> is the GABA inhibition coefficient
        if "GABA_SWITCH" in self.k_approch:
            linear_coef = 2 / self.linear_coef_threshold
            gaba_reponse = torch.minimum(
                torch.ones_like(self.neuron_activation_counters),
                torch.maximum(
                    -1 + (linear_coef * self.neuron_activation_counters),
                    torch.ones_like(self.neuron_activation_counters) * -1,
                )
            ).type_as(x)
        else:
            gaba_reponse = 1

        if self.use_mask:
            top_k_mask = torch.zeros_like(x)
            top_k_mask = top_k_mask.scatter(-1, inds, 1)
            x = x * top_k_mask
        else:
            x = self.Relu(x - (gaba_reponse * inhib_sig))
            if training:  # -> only append the activation cnts in training
                self.neuron_activation_counters += torch.sum(x > 0, dim=0, keepdim=True)

        return x


