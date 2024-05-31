from typing import Callable, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from spikingjelly.activation_based import learning, base, neuron, monitor, layer, encoding
from spikingjelly.activation_based.learning import stdp_linear_single_step, stdp_conv2d_single_step, stdp_multi_step


##############################################################
# Custom-written modules
class NonSpikingIFNode(neuron.IFNode):
    def forward(self, dv: torch.Tensor):
        if self.step_mode == 's':
            self.v_float_to_tensor(dv)
            self.neuronal_charge(dv)
            return self.v
        elif self.step_mode == 'm':
            T = dv.shape[0]
            v_seq = []
            for t in range(T):
                self.v_float_to_tensor(dv[t])
                self.neuronal_charge(dv[t])
                v = self.v
                v_seq.append(v)
            return torch.stack(v_seq)


def encoder():
    return encoding.PoissonEncoder()


##############################################################
# Custom-written modules
class Identity(nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__+'()'
        return tmpstr


class Reshape(nn.Module):
    def __init__(self, image_channels):
        super().__init__()
        self.image_channels = image_channels

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        image_size = int(np.sqrt(x.nelement() / (batch_size * self.image_channels)))
        return x.view(batch_size, self.image_channels, image_size, image_size)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + f"(channels={self.image_channels})"
        return tmpstr


##############################################################
# 通过设置spikingjelly里面的STDPLearner.step(on_grad=False)就可以获得原本的local plasticity的变化量，因此不需要单独再写一类
# 获得的delta_w应该是可以直接加到权重上的，不过可能线性连接不会直接采用裸权重就是了...
class Neuron_STDPLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str,
        sn_1: neuron.BaseNode, sn_2: neuron.BaseNode, synapse: Union[nn.Conv2d, nn.Linear],
        tau_pre: float, tau_post: float,
        w_min: float = -1., w_max: float = 1.,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
    ):
        super(Neuron_STDPLearner, self).__init__()
        self.step_mode = step_mode
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_min = w_min
        self.w_max = w_max
        self.f_pre = f_pre
        self.f_post = f_post
        self.synapse = synapse

        self.is_recurrent = True if sn_1 == sn_2 or id(sn_1)==id(sn_2) else False

        self.pre_spike_monitor = monitor.OutputMonitor(sn_1)
        self.post_spike_monitor = monitor.OutputMonitor(sn_2)

        self.register_memory('trace_pre', None)
        self.register_memory('trace_post', None)

    def reset(self):
        super(Neuron_STDPLearner, self).reset()
        self.pre_spike_monitor.clear_recorded_data()
        self.post_spike_monitor.clear_recorded_data()

    def disable(self):
        self.pre_spike_monitor.disable()
        self.post_spike_monitor.enable()

    def enable(self):
        self.pre_spike_monitor.enable()
        self.post_spike_monitor.enable()

    def step(self, on_grad=False, scale: float = 1.):
        length = self.pre_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == 's':
            if isinstance(self.synapse, nn.Conv2d):
                stdp_f = stdp_conv2d_single_step
            elif isinstance(self.synapse, nn.Linear):
                stdp_f = stdp_linear_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == 'm':
            if (isinstance(self.synapse, nn.Conv2d) or
                    isinstance(self.synapse, nn.Linear)):
                stdp_f = stdp_multi_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)
        with torch.no_grad():
            for _ in range(length):
                in_spike = self.pre_spike_monitor.records.pop(0)
                out_spike = self.post_spike_monitor.records.pop(0)

                self.trace_pre, self.trace_post, dw = stdp_f(
                    self.synapse, in_spike, out_spike,
                    self.trace_pre, self.trace_post,
                    self.tau_pre, self.tau_post,
                    self.f_pre, self.f_post
                )
                if scale != 1.:
                    dw *= scale

                delta_w = dw if (delta_w is None) else (delta_w + dw)

            # 自连接时消除神经元自己对自己的连接
            if self.is_recurrent:
                delta_w = delta_w - (torch.diag_embed(torch.diag(delta_w)))

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w


class Neuron_HebbLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str,
        sn_1, sn_2, synapse: Union[nn.Conv2d, nn.Linear],
    ):
        super(Neuron_HebbLearner, self).__init__()
        self.step_mode = step_mode
        self.synapse = synapse

        self.is_recurrent = True if sn_1 == sn_2 or id(sn_1)==id(sn_2) else False

        self.pre_spike_monitor = monitor.OutputMonitor(sn_1)
        self.post_spike_monitor = monitor.OutputMonitor(sn_2)

    def reset(self):
        super(Neuron_HebbLearner, self).reset()
        self.pre_spike_monitor.clear_recorded_data()
        self.post_spike_monitor.clear_recorded_data()

    def disable(self):
        self.pre_spike_monitor.disable()
        self.post_spike_monitor.disable()

    def enable(self):
        self.pre_spike_monitor.enable()
        self.post_spike_monitor.enable()

    def step(self, on_grad=False, scale: float = 1.):
        length = self.pre_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == 's':
            if isinstance(self.synapse, nn.Linear):
                stdp_f = hebb_linear_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == 'm':
            if isinstance(self.synapse, nn.Linear):
                stdp_f = hebb_multi_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        with torch.no_grad():
            for _ in range(length):
                in_spike = self.pre_spike_monitor.records.pop(0)
                out_spike = self.post_spike_monitor.records.pop(0)

                dw = stdp_f(self.synapse, in_spike, out_spike,)
                if scale != 1.:
                    dw *= scale

                delta_w = dw if (delta_w is None) else (delta_w + dw)

            # 自连接时消除神经元自己对自己的连接
            if self.is_recurrent:
                delta_w = delta_w - (torch.diag_embed(torch.diag(delta_w)))

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w


class Neuron_OjaLearner(base.MemoryModule):
    def __init__(
        self, step_mode: str,
        sn_1: neuron.BaseNode, sn_2: neuron.BaseNode, synapse: Union[nn.Conv2d, nn.Linear],
    ):
        super(Neuron_OjaLearner, self).__init__()
        self.step_mode = step_mode
        self.synapse = synapse

        self.is_recurrent = True if sn_1 == sn_2 or id(sn_1)==id(sn_2) else False

        self.pre_spike_monitor = monitor.OutputMonitor(sn_1)
        self.post_spike_monitor = monitor.OutputMonitor(sn_2)

    def reset(self):
        super(Neuron_OjaLearner, self).reset()
        self.pre_spike_monitor.clear_recorded_data()
        self.post_spike_monitor.clear_recorded_data()

    def disable(self):
        self.pre_spike_monitor.disable()
        self.post_spike_monitor.disable()

    def enable(self):
        self.pre_spike_monitor.enable()
        self.post_spike_monitor.enable()

    def step(self, on_grad=False, scale: float = 1.):
        length = self.pre_spike_monitor.records.__len__()
        delta_w = None

        if self.step_mode == 's':
            if isinstance(self.synapse, nn.Linear):
                stdp_f = oja_linear_single_step
            else:
                raise NotImplementedError(self.synapse)
        elif self.step_mode == 'm':
            if isinstance(self.synapse, nn.Linear):
                stdp_f = oja_multi_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)

        with torch.no_grad():
            for _ in range(length):
                in_spike = self.pre_spike_monitor.records.pop(0)
                out_spike = self.post_spike_monitor.records.pop(0)

                dw = stdp_f(self.synapse, in_spike, out_spike,)
                if scale != 1.:
                    dw *= scale

                delta_w = dw if (delta_w is None) else (delta_w + dw)

            # 自连接时消除神经元自己对自己的连接
            if self.is_recurrent:
                delta_w = delta_w - (torch.diag_embed(torch.diag(delta_w)))

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad = self.synapse.weight.grad - delta_w
        else:
            return delta_w


class ScaledWSLinear(layer.Linear):
    def __init__(self, in_features, out_features, bias=False, eps=1e-8):
        super(ScaledWSLinear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.eps = eps

    def get_weight(self):
        element_n = np.prod(self.weight.shape)
        mean = torch.mean(self.weight)
        var = torch.var(self.weight)
        weight = (self.weight - mean) / ((var * 784 + self.eps) ** 0.5)

        return weight

    def forward(self, x):
        return F.linear(x, self.get_weight(), self.bias)


class Flatten(nn.Module):
    """A module to flatten a multi-dimensional tensor to 2-dim tensor"""
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        return x.view(batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


class Flatten_SNN(nn.Module):
    """A module to flatten a multi-dimensional spiking tensor to 3-dim tensor"""
    def forward(self, x: torch.Tensor):
        time_step, batch_size = x.size(0), x.size(1)
        return x.view(time_step, batch_size, -1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'
        return tmpstr


def hebb_linear_single_step(
    fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
):
    in_sparse = torch.mean(in_spike, dim=1, keepdim=True)
    out_sparse = torch.mean(out_spike, dim=1, keepdim=True)
    delta_w = ((out_spike - out_sparse).unsqueeze(2) * (in_spike - in_sparse).unsqueeze(1)).sum(0)
    return delta_w


def oja_linear_single_step(
    fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
):
    delta_w = out_spike.unsqueeze(2) * (in_spike.unsqueeze(1) - out_spike.unsqueeze(2) * fc.weight.data)
    delta_w = delta_w.sum(0)
    return delta_w


def hebb_multi_step(
    layer: Union[nn.Linear, nn.Conv1d, nn.Conv2d],
    in_spike: torch.Tensor, out_spike:torch.Tensor,
):
    weight = layer.weight.data
    delta_w = torch.zeros_like(weight)
    T = in_spike.shape[0]

    if isinstance(layer, nn.Linear):
        hebb_single_step = hebb_linear_single_step
    else:
        raise NotImplementedError(layer)

    for t in range(T):
        dw = hebb_single_step(layer, in_spike[t], out_spike[t])
        delta_w += dw

    return delta_w


def oja_multi_step(
    layer: Union[nn.Linear, nn.Conv1d, nn.Conv2d],
    in_spike: torch.Tensor, out_spike:torch.Tensor,
):
    weight = layer.weight.data
    delta_w = torch.zeros_like(weight)
    T = in_spike.shape[0]

    if isinstance(layer, nn.Linear):
        hebb_single_step = oja_linear_single_step
    else:
        raise NotImplementedError(layer)

    for t in range(T):
        dw = hebb_single_step(layer, in_spike[t], out_spike[t])
        delta_w += dw

    return delta_w


def f_pre(x, w_min=-1., alpha=0.):
    return (x - w_min) ** alpha


def f_post(x, w_max=1., alpha=0.):
    return (w_max - x) ** alpha


def f_nan(x, w_min=-1.):
    return 1.

