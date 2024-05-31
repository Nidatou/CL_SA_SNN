import abc
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.utils import get_data_loader

from spikingjelly.activation_based import functional
from models.fc.layers import Gated_Layer
from models.utils.loss_func import softmax_cross_entropy


class CLBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(CLBase, self).__init__()
        self.grad_clip = 1.0

        # XdG
        self.mask_dict = None  # the mask for each hidden fully-connected layer
        self.gate_buffer_list = []  # <list> with the excite-buffers for all hidden layers

        # SI
        self.si_c = 0  # the strength to weigh SI-loss
        self.epsilon = 0.1  # dampening parameter: bounds 'omega' when squared parameter-change goes to 0

        # EWC
        self.ewc_lambda = 0  # the strength to weigh EWC-loss
        self.gamma = 1.  # decay-term for old tasks to quadractic term
        self.online = True   # online or offline for EWC
        self.fisher_n = None  # number batches for estimate
        self.EWC_task_count = 0  # keep track of number of quadratic loss terms

        # added parameter for experiment
        self.ewc_beta = 1.
        self.cl_batch_size = 512
        self.cl_batches = 5
        self.simpled_EWC = False

        # MAS
        self.mas_lambda = 0
        self.online_mas = True
        self.MAS_task_count = 0

        # Replay
        self.replay_targets = 'hard'  # should distillation loss to be used?
        self.KD_temp = 2.  # temp for distillation loss

    def _device(self):
        return next(self.parameters()).device

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, **kwargs):
        pass

    ######################################################################
    # XdG relevant functions
    def define_XdGmask(self, gating_prop, n_tasks):
        """Define the task-specific masks, by randomly selecting [gating_prop]% of nodes for fc-layers

        :param gating_prop: <num>, the proportion of nodes to be gated(0)
        :param n_tasks: <int> total number of tasks
        """
        mask_dict = {}
        gate_buffer_list = []
        for task_id in range(n_tasks):
            mask_dict[task_id + 1] = {}
            gate_ind = 0
            for layer_id in range(self.fcE.layers):
                layer = getattr(self.fcE, f"fcLayer{layer_id + 1}").gate_layer
                if task_id == 0:
                    gate_buffer_list.append(layer.gate_buffer)
                n_units = layer.gate_buffer.shape[0]
                gate_units = np.random.choice(n_units, size=int(gating_prop * n_units), replace=False)
                mask_dict[task_id+1][gate_ind] = gate_units
                gate_ind += 1

        self.mask_dict = mask_dict
        self.gate_buffer_list = gate_buffer_list

    # apply task-specific mask, by setting activity of pre-selected subset nodes to zeros
    def apply_XdGmask(self, task):
        assert self.mask_dict is not None
        torchType = next(self.parameters()).detach()

        for i, gate_buffer in enumerate(self.gate_buffer_list):
            gating_mask = np.repeat(1, gate_buffer.shape[0])
            gating_mask[self.mask_dict[task][i]] = 0.  # set the according gate
            gate_buffer.set_(torchType.new(gating_mask))  # apply the mask

    def reset_XdGmask(self):
        torchType = next(self.parameters()).detach()

        for i, gate_buffer in enumerate(self.gate_buffer_list):
            gating_mask = np.repeat(1, gate_buffer.shape[0])
            gate_buffer.set_(torchType.new(gating_mask))

    ######################################################################
    # EWC relevant functions
    def estimate_fisher(self, dataset, allowed_classes=None):
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        # Set model to evaluate mode, but this can not be used in spikingjelly
        # mode = self.training
        # self.train()
        # self.eval()

        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda())

        if not self.simpled_EWC:
            for index, (x, y) in enumerate(data_loader):
                if self.fisher_n is not None:
                    if index >= self.fisher_n:
                        break

                x = x.to(self._device())
                output = self(x, training=False) if allowed_classes is None else self(x, training=False)[:, allowed_classes]
                functional.reset_net(self)
                with torch.no_grad():
                    label_weights = F.softmax(output, dim=1)
                for label_index in range(output.shape[1]):
                    label = torch.LongTensor([label_index]).to(self._device())
                    negloglikelihood = F.cross_entropy(output, label)
                    # Calculate gradient of negative loglikelihood
                    self.zero_grad()
                    negloglikelihood.backward(retain_graph=True if (label_index + 1) < output.shape[1] else False)

                    for n, p in self.named_parameters():
                        if p.requires_grad:
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)
            # Normalize bt sample size used for estimation
            est_fisher_info = {n: p / index for n, p in est_fisher_info.items()}
        else:
            data_loader = get_data_loader(dataset, batch_size=self.cl_batch_size, cuda=self._is_on_cuda())
            for index, (x, y) in enumerate(data_loader):
                if index > self.cl_batches:
                    break
                x = x.to(self._device())
                output = self(x, training=False)
                pred_label = output.max(1)[1].flatten()

                loss = softmax_cross_entropy(output, pred_label, self.ewc_beta)
                self.zero_grad()
                loss.backward()
                functional.reset_net(self)
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher_info[n] += (p.grad.detach() ** 2) / (self.cl_batches * self.cl_batch_size)

        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer(f'{n}_EWC_prev_task{"" if self.online else self.EWC_task_count+1}',
                                     p.detach().clone())
                # -precision (approximated bt diagonmal Fisher Information matrix)
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self, f'{n}_EWC_estimated_fisher')
                    est_fisher_info[n] += self.gamma * existing_values
                self.register_buffer(f'{n}_EWC_estimated_fisher{"" if self.online else self.EWC_task_count+1}',
                                     est_fisher_info[n])

        # if offline, increase task-count (and set it to 1 to indicate EWC_loss can be calculated)
        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        # self.train(mode=mode)

    # Calculate EWC-loss
    def ewc_loss(self):
        if self. EWC_task_count > 0:
            losses = []
            for task_id in range(1, self.EWC_task_count + 1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        mean = getattr(self, f"{n}_EWC_prev_task{''if self.online else task_id}")
                        fisher = getattr(self, f"{n}_EWC_estimated_fisher{'' if self.online else task_id}")
                        fisher = self.gamma * fisher if self.online else fisher
                        losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2) * sum(losses)

        else:
            # when EWC_task_count is 0, it means there is no Fisher right now
            return torch.tensor(0., device=self._device())

    ######################################################################
    # SI(synapse Intelligence) relevant functions
    # update the per-parameter
    def update_omega(self, W, epsilon):
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                p_prev = getattr(self, f'{n}_SI_prev_task')
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n]/(p_change**2 + epsilon)
                try:
                    omega = getattr(self, f"{n}_SI_omega")
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.register_buffer(f'{n}_SI_prev_task', p_current)
                self.register_buffer(f'{n}_SI_omega', omega_new)

    # calculate SI's surrogate loss
    def surrogate_loss(self):
        try:
            losses = []
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    prev_value = getattr(self, f'{n}_SI_prev_task')
                    omega = getattr(self, f'{n}_SI_omega')
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p-prev_value) ** 2).sum())
            return sum(losses)
        except AttributeError:
            return torch.tensor(0., device=self._device())

    ######################################################################
    # MAS relevant functions
    def estimate_mas_importance(self, dataset, allowed_classes=None):
        mas_info = {}
        # print('mas importance calculated')
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                mas_info[n] = p.detach().clone().zero_()

        data_loader = get_data_loader(dataset, batch_size=self.cl_batch_size, cuda=self._is_on_cuda())

        for index, (x, y) in enumerate(data_loader):
            if index > self.cl_batches:
                break
            x = x.to(self._device())
            pred = self(x, training=False)
            pred.pow_(2)
            loss = pred.mean()

            self.zero_grad()
            loss.backward()
            functional.reset_net(self)
            for n, p in self.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        mas_info[n] += (p.grad.detach().abs() / len(data_loader))

        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer(f"{n}_MAS_prev_task{'' if self.online_mas else self.MAS_task_count+1}",
                                     p.detach().clone())
                if self.online_mas and self.MAS_task_count == 1:
                    existing_values = getattr(self, f'{n}_MAS_INFO')
                    mas_info[n] += existing_values
                self.register_buffer(f"{n}_MAS_INFO{'' if self.online_mas else self.MAS_task_count+1}",
                                     mas_info[n])
        self.MAS_task_count = 1 if self.online_mas else self.MAS_task_count + 1

        self.train()

    def mas_loss(self):
        if self.MAS_task_count > 0:
            losses = []
            for task_id in range(1, self.MAS_task_count + 1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        mean = getattr(self, f"{n}_MAS_prev_task{'' if self.online_mas else task_id}")
                        mas_info = getattr(self, f"{n}_MAS_INFO{'' if self.online_mas else task_id}")
                        losses.append((mas_info * (p - mean) ** 2).sum())
                        # Sum MAS-loss from all parameters (and from all tasks, if "offline EWC")
            return (1. / 2) * sum(losses)

        else:
            return torch.tensor(0., device=self._device())





