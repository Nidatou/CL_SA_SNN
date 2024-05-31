import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from spikingjelly.activation_based import base, neuron, surrogate, layer, functional


#######################################################################
# several solutions for kWTA for spike-version's WTA
class kWTA_Spike(nn.Module):
    def __init__(
        self, neu_size, device=None, neuron_type="if", surr_func=surrogate.Sigmoid(),
        # -> parameter for specific excite and inhibit
        last_inhibit=False, curr_excite=False, tune_approach="sigmoid", inhibit_p=1e-4, excite_p=1e-4,
        tune_param=1., adapt_tune=True,
        # -> parameter for adaptive threshold
        adaptive_threshold=False, adap_approach="linear", adap_param=100000, thresh_min=1., thresh_max=2.,
        # -> parameter for the k's choice
        k_min=10, soft_reset=True, detach_nonspike=False, k_param=50, k_approach="FLAT", k_max=None,
    ):
        super(kWTA_Spike, self).__init__()
        v_reset = None if soft_reset else 0.
        if neuron_type == 'if':
            self.neu = Adapt_Thresh_IFNode(v_reset=v_reset, v_threshold=thresh_min, surrogate_function=surr_func)
        elif neuron_type == 'lif':
            self.neu = Adapt_Thresh_LIFNode(v_reset=v_reset, v_threshold=thresh_min, surrogate_function=surr_func)
        else:
            raise NotImplementedError("Now only support IF <if> and LIF <lif> neurons")
        self.neu_size = neu_size

        # the parameters concerning about the task-specific WTA algorithm
        self.last_inhibit = last_inhibit  # -> whether the neurons activated in last will inhibit in this task
        self.curr_excite = curr_excite  # -> whether the neurons activated now will more possibly to activate
        self.inhibit_p = inhibit_p if last_inhibit else 0
        self.excite_p = excite_p if curr_excite else 0
        self.tune_param = tune_param  # -> decide the spiking tuning scale
        self.adapt_tune = adapt_tune  # -> whether to tune the k
        if tune_approach == "sigmoid":
            self.act_tune = self.spike_sigmoid
        elif tune_approach == "tanh":
            self.act_tune = self.spike_tanh
        elif tune_approach == "linear":
            self.act_tune = self.spike_linear
        else:
            raise NotImplementedError(f"don't have the activation tunning function{tune_approach}")

        # -> counters for activation in tasks before
        self.last_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)
        self.neu_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)
        self.task_num = 1

        # some parameters for k_approach
        self.k_approach = k_approach
        assert self.k_approach in ['LINEAR_DECAY', 'FLAT']
        self.k_min = k_min
        self.k_max = self.neu_size if k_max is None else k_max
        self.k_param = k_param

        # some parameters for adaptive_threshold
        self.detach_nonspike = detach_nonspike
        self.adaptive_threshold = adaptive_threshold
        if self.adaptive_threshold:
            if isinstance(self.neu.v_threshold, float):
                self.neu.v_threshold = torch.full((neu_size,), self.neu.v_threshold).to(device)
            self.adap_param = adap_param
            assert adap_approach in ['linear', 'exponential']
            self.thresh_min = thresh_min
            self.thresh_max = thresh_max
            self.adap_approach = adap_approach

        #################################
        # generate a name for the model #
        #################################
        specific_label = "{excite}{inhibit}{scale}".format(
            inhibit="" if not self.last_inhibit else "-inh",
            excite="" if not self.curr_excite else f"{'-aexc' if self.adapt_tune else '-exc'}",
            scale=f"s{self.tune_param}" if self.last_inhibit or self.curr_excite else "",
        )
        neu_label = "-{neu_type}{threshold}".format(
            neu_type=neuron_type,
            threshold="" if not self.adaptive_threshold
            else "_{approach}{scale}{parameter}".format(
                approach="l" if self.adap_approach=="linear" else "ex",
                scale=f"{self.thresh_min}_{self.thresh_max}",
                parameter=f"_p{int(self.adap_param)}",
            )
        )
        k_label = "-{approach}{parameters}".format(
            approach="F_" if self.k_approach == "FLAT" else "LD_",
            parameters=f"{self.k_min}" if self.k_approach == "FLAT"
            else f"{self.k_min}-{self.k_max}-p{int(self.k_param)}"
        )
        self.label = f"-kspk{neu_label}{k_label}{specific_label}"

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
            k = int(k+0.5)
        else:
            k = self.k_min

        return k

    def forward(self, x: torch.Tensor, curr_ep=None, training=True):
        # -> kWTA for Spike can only used when finished the whole time window
        assert self.neu.step_mode == 'm' and len(x.shape) == 3
        spike_out = self.neu(x)
        if training:
            sum_spike = self.act_tune(torch.sum(spike_out.detach(), dim=0))
        else:
            sum_spike = torch.sum(spike_out.detach(), dim=0)
        curr_k = self.get_curr_k(curr_ep=curr_ep)

        vals, inds = torch.topk(sum_spike, curr_k, dim=-1, sorted=False)
        inhib_sig, _ = torch.min(vals.detach(), dim=-1, keepdim=True)

        gaba_reponse = 1  # -> store a value for preparation
        top_k_mask = torch.zeros_like(sum_spike)
        top_k_mask = top_k_mask.scatter(-1, inds, 1)
        spike_out = spike_out * top_k_mask

        if training:
            self.neu_act_cnts += torch.sum(torch.sum(spike_out, dim=0, keepdim=True), dim=1).detach()
        if self.adaptive_threshold:
            if self.adap_approach == "linear":
                raw_thresh = self.thresh_min + (self.neu_act_cnts + self.last_act_cnts) / self.adap_param
            elif self.adap_approach == "exponential":
                raw_thresh = self.thresh_min + (self.thresh_max - self.thresh_min) * (
                            1. - torch.exp(-(self.neu_act_cnts + self.last_act_cnts) * 2 / self.adap_param))
            else:
                raise NotImplementedError("only support <linear> and <exponential> adaptive threshold")
            self.neu.v_threshold = torch.minimum(
                self.thresh_max * torch.ones_like(self.neu.v_threshold),
                torch.maximum(
                    torch.ones_like(self.neu.v_threshold) * self.thresh_min,
                    raw_thresh,
                )
            )

        if self.detach_nonspike:
            spike_out = spike_out * spike_out.detach()
        return spike_out

    # -> accumulate the spike sum once a task is ended
    def end_one_task(self):
        self.last_act_cnts += self.neu_act_cnts
        self.neu_act_cnts = torch.zeros_like(self.neu_act_cnts)
        if self.adapt_tune:
            self.excite_p = self.excite_p * (self.task_num + 1) / self.task_num
            self.task_num += 1

    def spike_sigmoid(self, sum_spike):
        summed_param = self.neu_act_cnts * self.excite_p - self.last_act_cnts * self.inhibit_p
        return sum_spike * (torch.sigmoid(summed_param) * self.tune_param + (1. - self.tune_param/2))

    def spike_tanh(self, sum_spike):
        summed_param = self.neu_act_cnts * self.excite_p - self.last_act_cnts * self.inhibit_p
        return sum_spike * (1 + torch.tanh(summed_param) * self.tune_param / 2)

    def spike_linear(self, sum_spike):
        summed_param = self.neu_act_cnts * self.excite_p - self.last_act_cnts * self.inhibit_p
        summed_param = torch.minimum(
            torch.ones_like(self.last_act_cnts) * (1. + self.tune_param/2),
            torch.maximum(
                torch.ones_like(self.last_act_cnts) * (1. - self.tune_param/2),
                summed_param + 1,
            )
        )
        return sum_spike * summed_param


# 我先自己构思一下这种通过膜电位进行K-WTA的构思，上面的Spike模式只能用mask实在很让人担心
# 上面的方法显然只能采用多步采样+mask的方法，膜电位的方法注重单步更新，但是也能多步
# <use_mask> 对于没能WTA未能选中的神经元的膜电压应作何处理(mask/substract)
# <soft_reset> 神经元应该如何在释放脉冲以后reset
# <single-WTA> 是否在每个时间步都硬性采用WTA，还是实际上在“某个范围”内采用WTA
# <...> 是否要采取某种策略防止某些神经元死了
# (写的时候自己也不抱什么信心太草了)
class kWTA_Mem(neuron.BaseNode):
    def __init__(
        self, surr_func=surrogate.Sigmoid(), neu_size=None, neu_type='if', device=None,
        # -> parameter for k-WTA
        use_mask=True, soft_reset=True, single_WTA=False, k_min=10, detach_nonspike=False,
        adaptive_threshold=False, adap_approach="linear", adap_param=None, inhibit_intense=0.1,
        thresh_min=0.5, thresh_max=1.
    ):
        assert neu_size is not None
        v_reset = None if soft_reset else 0.
        self.tau = 2
        super(kWTA_Mem, self).__init__(v_reset=v_reset, v_threshold=thresh_min, surrogate_function=surr_func)
        self.neu_size = neu_size
        self.neu_type = neu_type
        self.use_mask = use_mask
        self.single_WTA = single_WTA
        self.k_min = k_min
        self.detach_nonspike = detach_nonspike
        self.adaptive_threshold = adaptive_threshold if not single_WTA else False

        self.neuron_activation_counters = torch.zeros((1, self.neu_size), requires_grad=False).to(device)

        self.v_threshold = torch.full((neu_size,), self.v_threshold).to(device)
        if not self.single_WTA:
            self.inhibit_matrix = torch.ones((self.neu_size, self.neu_size)).to(device)
            inhibit_intense = 1 / self.k_min if inhibit_intense is None else inhibit_intense
            self.inhibit_matrix *= inhibit_intense
            for i in range(self.neu_size):
                self.inhibit_matrix[i, i] = 0.
            if adaptive_threshold:
                self.thresh_max = thresh_max
                self.thresh_min = thresh_min
                self.adap_param = adap_param
                self.adap_approach = adap_approach
        if self.neu_type == 'if':
            self.charge_func = self.if_neuron_charge
        elif self.neu_type == 'lif':
            self.charge_func = self.lif_neuron_charge
        else:
            raise NotImplementedError("only have charge function")

        # self.tau = 2.
        # self.alpha = 1.
        # self.w_min = 0
        # self.w_max = 2
        # self.f_pre = lambda x, w_min=self.w_min, alpha=self.alpha: (x-w_min)**alpha
        # self.f_post = lambda x, w_max=self.w_max, alpha=self.alpha: (w_max-x) ** alpha
        # self.spike_trace = None
        # self.stdp_lr = 1e-2

        #################################
        # generate a name for the model #
        #################################
        specific_label = "-single" if self.single_WTA else ""
        neu_label = "{neu_type}{threshold}".format(
            neu_type=self.neu_type,
            threshold="" if not self.adaptive_threshold
            else "-{approach}{scale}{parameter}".format(
                approach="l" if self.adap_approach == "linear" else "ex",
                scale=f"{self.thresh_min}_{self.thresh_max}",
                parameter=f"_p{int(self.adap_param)}",
            )
        )
        k_label = "-{approach}{parameters}".format(
            approach="", parameters=f"k{self.k_min}",
        )
        self.label = f"-kmem{neu_label}{k_label}{specific_label}"

    @property
    def name(self):
        return self.label

    def if_neuron_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def lif_neuron_charge(self, x: torch.Tensor):
        # 如果想要使用lif-neuron的话就必须将阈值上限设置为1，但是刚开始写的时候并没有注意，
        # 所以现在没法观察了（不过估计神经元类型也不会产生太大影响就是了
        self.v = self.v + (x - self.v) / self.tau
        # self.v = self.v * (1. - 1. / self.tau) + x

    def get_curr_k(self, curr_ep=None):
        k = self.k_min
        return k

    def top_k_fire(self, curr_ep=None):
        curr_k = self.get_curr_k(curr_ep=curr_ep)
        if self.single_WTA:  # -> just use single k-WTA for every time step

            vals, inds = torch.topk(self.v, curr_k, dim=-1, sorted=False)
            inhib_sig, _ = torch.min(vals.detach(), dim=-1, keepdim=True)

            gaba_reponse = 1
            spike = self.surrogate_function(self.v - inhib_sig)
            self.v -= inhib_sig
        else:
            # ->如果在保持阈值的前提下仍然要在单步也使用mask，那就会出现刚发放过脉冲的神经元在下一个时间步被快速保持在零
            # ->反而失去了k-WTA的能力，所以一时间想不到该怎么排除这种问题，就先不考虑
            spike = self.surrogate_function(self.v - self.v_threshold)
            self.v -= torch.matmul(spike, self.inhibit_matrix).detach()

            spike_d = spike.detach() if self.detach_reset else spike
            if self.v_reset is None:
                self.v = self.v - spike_d * self.v_threshold
            else:
                self.v = (1. - spike_d) * self.v + spike * self.v_reset

            # 也许还涉及到对脉冲阈值的调整
            self.neuron_activation_counters += torch.sum(spike, dim=0, keepdim=True).detach()
            if self.adaptive_threshold:
                if self.adap_approach == 'linear':
                    raw_thresh = self.thresh_min + self.neuron_activation_counters / self.adap_param
                elif self.adap_approach == "exponential":
                    raw_thresh = self.thresh_min + (self.thresh_max - self.thresh_min) * (
                            1. - torch.exp(-self.neuron_activation_counters * 2 / self.adap_param))
                else:
                    raise NotImplementedError("only support <linear> and <exponential> for adaptive threshold")
                self.v_threshold = torch.minimum(
                    torch.ones_like(self.v_threshold) * self.thresh_max,
                    torch.maximum(
                        torch.ones_like(self.v_threshold) * self.thresh_min,
                        raw_thresh,
                    )
                )
        if self.detach_nonspike:
            spike = spike * spike.detach()
        return spike

    def single_step_forward(self, x: torch.Tensor, curr_ep=None, **kwargs):
        self.v_float_to_tensor(x)
        self.charge_func(x)
        spike = self.top_k_fire(curr_ep=curr_ep)
        # 注意还要考虑一下每次初始化的问题
        # self.stdp_func(spike.detach())
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor, curr_ep=None, **kwargs):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def end_one_task(self):
        # -> nothing need to do for this...
        return None

    # def stdp_func(self, spike):
    #     assert not self.single_WTA
    #     if self.spike_trace is None:
    #         self.spike_trace = 0.
    #     trace_spike = self.spike_trace - self.spike_trace / self.tau + spike
    #     delta_w_pre = -self.f_pre(self.inhibit_matrix) * (trace_spike.unsqueeze(2) * spike.unsqueeze(1)).sum(0)
    #     delta_w_post = self.f_post(self.inhibit_matrix) * (trace_spike.unsqueeze(1) * spike.unsqueeze(2)).sum(0)
    #     delta_w = delta_w_pre + delta_w_post
    #     self.inhibit_matrix += delta_w.T
    #     for i in range(self.neu_size):
    #         self.inhibit_matrix[i, i] = 0.


# 延时间步版本的snn-kwta，虽然又是一个意义不明的设想...
# 基本思路大概就是网络维持一个trace用来记录每个神经元的激活轨迹（会衰减），然后根据这个轨迹来在每个时间步进行k的筛选
# （毕竟SNN每个时间步只有1、0做不了ANN的实数筛选）
# 对于这个轨迹有两种思路，一种是只在单个batch中衰减（也就是过了batch就重置），一种是在整个任务结束以后才重置
# 但这样又有个问题...就是一组batch并不止一个数据，如果考虑单个任务的情况，就不得不考虑是不是只保留一组trace
# 当然，只要神经元达到了脉冲条件就会对trace产生影响，而与它们是否成为k并没有什么关系
# 同样的问题还有所谓的抑制...不过仔细想想似乎还是上面那个问题更加恼人一点
class kWTA_Spk_tr(neuron.BaseNode):
    def __init__(
        self, surr_func=surrogate.Sigmoid(), neu_size=None, neu_type='if', device=None,
        # -> parameter for k-WTA
        soft_reset=True, k_min=10, inhibit_intense=1.0, use_inhibit=False, inhibit_self=False, hard_mask=True,
        k_approach="FLAT", k_param=20000, k_max=None,
        # -> parameter for trace
        tr_decay=10., loosen_tr=False,
        # -> parameters for adaptive threshold
        adaptive_threshold=False, adap_approach="linear", adap_param=None, thresh_min=1.0, thresh_max=2.0,
    ):
        assert neu_size is not None
        v_reset = None if soft_reset else 0.
        self.tau = 2.
        super(kWTA_Spk_tr, self).__init__(v_reset=v_reset, v_threshold=thresh_min, surrogate_function=surr_func)
        self.neu_size = neu_size
        self.neu_type = neu_type

        # -> parameter about inhibit
        self.hard_mask = hard_mask
        # self.use_inhibit = use_inhibit
        # self.inhibit_matrix = torch.ones((self.neu_size, self.neu_size)).to(device) * inhibit_intense
        # if not inhibit_self:
        #     for i in range(self.neu_size):
        #         self.inhibit_matrix[i, i] = 0.

        # -> parameter about trace
        self.neu_tr = None
        self.tr_decay = tr_decay
        self.loos_tr = loosen_tr

        # ->parameter about adaptive threshold
        self.adaptive_threshold = adaptive_threshold
        self.last_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)
        self.neu_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)

        if adaptive_threshold:
            self.thresh_max = thresh_max
            self.thresh_min = thresh_min
            self.adap_param = adap_param
            self.adap_approach = adap_approach
        self.v_threshold = torch.full((neu_size,), self.v_threshold).to(device)
        # -> parameter about neuron itself
        self.k_min = k_min
        assert k_approach in ['LINEAR_DECAY', 'FLAT']
        self.k_approach = k_approach
        self.k_param = k_param
        self.k_max = self.neu_size if k_max is None else k_max

        if self.neu_type == 'if':
            self.charge_func = self.if_neuron_charge
        elif self.neu_type == 'lif':
            self.charge_func = self.lif_neuron_charge
        else:
            raise NotImplementedError("only have charge function")
        #################################
        # generate a name for the model #
        #################################
        tr_label = "-{tr_size}_{tr_refresh}_{tr_decay}{inh}{loos}".format(
            tr_size="ba",
            tr_refresh="b-ref",
            tr_decay=f"d{self.tr_decay}",
            inh="" if not use_inhibit else f"{'_re-' if inhibit_self else ''}{inhibit_intense}",
            loos="" if not self.loos_tr else "loos",
        )
        neu_label = "{neu_type}{threshold}".format(
            neu_type=self.neu_type,
            threshold="" if not self.adaptive_threshold
            else "-{approach}{scale}{parameter}".format(
                approach="l" if self.adap_approach == "linear" else "ex",
                scale=f"{self.thresh_min}_{self.thresh_max}",
                parameter=f"_p{int(self.adap_param)}",
            )
        )
        k_label = "-{approach}{parameters}-{k_app}{k_param}".format(
            approach="hard_" if self.hard_mask else "soft_",
            parameters=f"k{self.k_min}",
            k_app="F_" if k_approach == "FLAT" else "LD_",
            k_param=f"{self.k_min}" if self.k_approach == "FLAT"
            else f"{self.k_min}-{self.k_max}-p{int(self.k_param)}"
        )
        self.label = f"-kspktr{neu_label}{k_label}{tr_label}"

    @property
    def name(self):
        return self.label

    def if_neuron_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def lif_neuron_charge(self, x: torch.Tensor):
        # 如果想要使用lif-neuron的话就必须将阈值上限设置为1，但是刚开始写的时候并没有注意，
        # 所以现在没法观察了（不过估计神经元类型也不会产生太大影响就是了
        # 另外一方面原因在于对第一层的输入进行了归一化导致输入值普遍比较小...
        self.v = self.v + (x - self.v) / self.tau
        # self.v = self.v * (1. - 1. / self.tau) + x

    def get_curr_k(self, curr_ep=None):
        if curr_ep == None:
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
            k = int(k+0.5)
        else:
            k = self.k_min
        return k

    def top_k_fire(self, curr_ep=None, training=True, final_step=False):
        curr_k = self.get_curr_k(curr_ep=curr_ep)
        # -> reset the neu_tr
        if self.neu_tr is None:
            self.neu_tr = torch.zeros_like(self.v).to(self.v.data.device)

        spike = self.surrogate_function(self.v - self.v_threshold)
        append_tr = spike.detach()
        if self.tr_decay != 0:
            self.neu_tr = self.neu_tr - self.neu_tr / self.tr_decay + append_tr
        else:
            self.neu_tr = self.neu_tr + append_tr

        # -> implement the WTA-k to the spike
        if self.hard_mask or not final_step:
            if self.loos_tr:
                vals, inds = torch.topk(self.neu_tr, curr_k, dim=-1, sorted=False)
                thresh, _ = torch.min(vals.detach(), dim=-1, keepdim=True)  # -> get the no.k's threshold
                top_k_mask = ((self.neu_tr >= thresh) & (self.neu_tr > 0)).float()
            else:
                vals, inds = torch.topk(self.neu_tr, curr_k + 1, dim=-1, sorted=False)
                thresh, _ = torch.min(vals.detach(), dim=-1, keepdim=True)  # -> get the no.k's threshold
                top_k_mask = (self.neu_tr > thresh).float()
        else:  # soft mask if not usable for CL, but we can only activate the last step's spike to at least make the learning convergence

            rand_ind = torch.randperm(self.neu_tr.shape[-1]).to(self.v.device)
            vals, inds = torch.topk(self.neu_tr[..., rand_ind], curr_k, dim=-1, sorted=False)
            thres, _ = torch.min(vals.detach(), dim=-1, keepdim=True)

            top_k_mask = torch.zeros_like(self.neu_tr)
            top_k_mask = top_k_mask.scatter(-1, rand_ind[inds], 1)
        # vals, inds = torch.topk(self.neu_tr, curr_k + 1, dim=-1, sorted=False)
        # thresh, _ = torch.min(vals.detach(), dim=-1, keepdim=True)  # -> get the no.k's threshold
        #
        # top_k_mask = (self.neu_tr > thresh).float() if self.hard_mask else (self.neu_tr >= thresh).float()

        spike_out = top_k_mask * spike

        # -> reset the mem of the neurons
        spike_d = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_d * self.v_threshold
        else:
            self.v = (1. - spike_d) * self.v + spike * self.v_reset

        # if self.use_inhibit:
        #     self.v -= torch.matmul(spike_out, self.inhibit_matrix).detach()

        # -> implement the adaptive threshold
        if training:
            self.neu_act_cnts += torch.sum(spike_out, dim=0, keepdim=True).detach()
        if self.adaptive_threshold:
            if self.adap_approach == 'linear':
                raw_thresh = self.thresh_min + (self.neu_act_cnts + self.last_act_cnts) / self.adap_param
            elif self.adap_approach == "exponential":
                raw_thresh = self.thresh_min + (self.thresh_max - self.thresh_min) * (
                        1. - torch.exp(-(self.neu_act_cnts + self.last_act_cnts) * 2 / self.adap_param))
            else:
                raise NotImplementedError("only support <linear> and <exponential> for adaptive threshold")
            self.v_threshold = torch.minimum(
                torch.ones_like(self.v_threshold) * self.thresh_max,
                torch.maximum(
                    torch.ones_like(self.v_threshold) * self.thresh_min,
                    raw_thresh,
                )
            )

        return spike_out

    def single_step_forward(self, x: torch.Tensor, curr_ep=None, training=True, final_step=False, **kwargs):
        assert self.step_mode == 'm', 'This topk node only support multi-step for convenience'
        self.v_float_to_tensor(x)
        self.charge_func(x)
        spike = self.top_k_fire(curr_ep=curr_ep, training=training, final_step=final_step)
        # 注意还要考虑一下每次初始化的问题
        # self.stdp_func(spike.detach())
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor, curr_ep=None, training=True, **kwargs):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            final_step = True if (t == T - 1) and torch.sum(torch.stack(y_seq)) < self.k_min else False
            y = self.single_step_forward(x_seq[t], training=training, final_step=final_step)
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        self.neu_tr = None

        return torch.stack(y_seq)

    def end_one_task(self):
        self.last_act_cnts += self.neu_act_cnts
        self.neu_act_cnts = torch.zeros_like(self.neu_act_cnts)
        return None


class kWTA_Mem_spk(neuron.BaseNode):
    def __init__(
        self, surr_func=surrogate.Sigmoid(), neu_size=None, neu_type='if', device=None,
        # -> parameter for k-WTA
        soft_reset=True, k_min=10,
        # -> parameters for adaptive threshold
        adaptive_threshold=False, adap_approach="linear", adap_param=None, thresh_min=1.0, thresh_max=2.0,
    ):
        assert neu_size is not None
        v_reset = None if soft_reset else 0.
        self.tau = 2.
        super(kWTA_Mem_spk, self).__init__(v_reset=v_reset, v_threshold=thresh_min, surrogate_function=surr_func)
        self.neu_size = neu_size
        self.neu_type = neu_type

        self.neu_tr = None

        # ->parameter about adaptive threshold
        self.adaptive_threshold = adaptive_threshold
        self.last_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)
        self.neu_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)

        if adaptive_threshold:
            self.thresh_max = thresh_max
            self.thresh_min = thresh_min
            self.adap_param = adap_param
            self.adap_approach = adap_approach
        self.v_threshold = torch.full((neu_size,), self.v_threshold).to(device)
        # -> parameter about neuron itself
        self.k_min = k_min

        if self.neu_type == 'if':
            self.charge_func = self.if_neuron_charge
        elif self.neu_type == 'lif':
            self.charge_func = self.lif_neuron_charge
        else:
            raise NotImplementedError("only have charge function")
        #################################
        # generate a name for the model #
        #################################

        neu_label = "{neu_type}{threshold}".format(
            neu_type=self.neu_type,
            threshold=f"{self.thresh_min}" if not self.adaptive_threshold
            else "-{approach}{scale}{parameter}".format(
                approach="l" if self.adap_approach == "linear" else "ex",
                scale=f"{self.thresh_min}_{self.thresh_max}",
                parameter=f"_p{int(self.adap_param)}",
            )
        )
        k_label = "-{parameters}".format(
            parameters=f"k{self.k_min}",
        )
        self.label = f"-kmem{neu_label}{k_label}"

    @property
    def name(self):
        return self.label

    def if_neuron_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def lif_neuron_charge(self, x: torch.Tensor):
        # 如果想要使用lif-neuron的话就必须将阈值上限设置为1，但是刚开始写的时候并没有注意，
        # 所以现在没法观察了（不过估计神经元类型也不会产生太大影响就是了
        # 另外一方面原因在于对第一层的输入进行了归一化导致输入值普遍比较小...
        self.v = self.v + (x - self.v) / self.tau
        # self.v = self.v * (1. - 1. / self.tau) + x

    def top_k_fire(self, curr_ep=None, training=True):
        curr_k = self.k_min
        # -> reset the neu_tr
        self.neu_tr = torch.zeros_like(self.v).to(self.v.data.device)

        spike = self.surrogate_function(self.v - self.v_threshold)
        self.neu_tr = self.neu_tr + spike.detach() * self.thresh_max
        # -> reset the mem of the neurons
        spike_d = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_d * self.v_threshold
        else:
            self.v = (1. - spike_d) * self.v + spike * self.v_reset

        # simply use membrane and the spike sum as the benchmarks
        # this seem to lose part of regularization function of the spike
        comp_refer = (self.neu_tr + self.v).detach()
        # comp_refer = ((self.neu_tr + self.v) * (self.neu_tr > 0).float()).detach()
        vals, inds = torch.topk(comp_refer, curr_k + 1, dim=-1, sorted=False)
        thresh, _ = torch.min(vals.detach(), dim=-1, keepdim=True)
        top_k_mask = (comp_refer > thresh).float()

        spike_out = top_k_mask * spike

        # -> implement the adaptive threshold
        if training:
            self.neu_act_cnts += torch.sum(spike_out, dim=0, keepdim=True).detach()
        if self.adaptive_threshold:
            if self.adap_approach == 'linear':
                raw_thresh = self.thresh_min + (self.neu_act_cnts + self.last_act_cnts) / self.adap_param
            elif self.adap_approach == "exponential":
                raw_thresh = self.thresh_min + (self.thresh_max - self.thresh_min) * (
                        1. - torch.exp(-(self.neu_act_cnts + self.last_act_cnts) * 2 / self.adap_param))
            else:
                raise NotImplementedError("only support <linear> and <exponential> for adaptive threshold")
            self.v_threshold = torch.minimum(
                torch.ones_like(self.v_threshold) * self.thresh_max,
                torch.maximum(
                    torch.ones_like(self.v_threshold) * self.thresh_min,
                    raw_thresh,
                )
            )

        return spike_out

    def single_step_forward(self, x: torch.Tensor, curr_ep=None, training=True, **kwargs):
        assert self.step_mode == 'm', 'This topk node only support multi-step for convenience'
        self.v_float_to_tensor(x)
        self.charge_func(x)
        spike = self.top_k_fire(curr_ep=curr_ep, training=training)
        # 注意还要考虑一下每次初始化的问题
        # self.stdp_func(spike.detach())
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor, curr_ep=None, training=True, **kwargs):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t], curr_ep=curr_ep, training=training)
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        self.neu_tr = None

        return torch.stack(y_seq)

    def end_one_task(self):
        self.last_act_cnts += self.neu_act_cnts
        self.neu_act_cnts = torch.zeros_like(self.neu_act_cnts)
        return None


# 再整一个Spk_tr也参与抑制/激活的版本
class kWTA_Spk_tr_inh(neuron.BaseNode):
    def __init__(
        self, surr_func=surrogate.Sigmoid(), neu_size=None, neu_type='if', device=None,
        # -> parameter for k-WTA
        soft_reset=True, k_min=10, hard_mask=True, tr_decay=10.,
        # -> parameter for specific excite and inhibit
        last_inhibit=False, curr_excite=False, inhibit_p=1e-6, excite_p=1e-4,
        tune_param=1., adapt_tune=True,
        # -> parameters for adaptive threshold
        adaptive_threshold=False, adap_approach="linear", adap_param=None, thresh_min=1.0, thresh_max=2.0,
    ):
        assert neu_size is not None
        v_reset = None if soft_reset else 0.
        self.tau = 2.
        super(kWTA_Spk_tr_inh, self).__init__(v_reset=v_reset, v_threshold=thresh_min, surrogate_function=surr_func)
        self.neu_size = neu_size
        self.neu_type = neu_type

        # -> parameter about task-specific WTA algorithm
        self.last_inhibit = last_inhibit
        self.curr_excite = curr_excite
        self.inhibit_p = inhibit_p if last_inhibit else 0
        self.excite_p = excite_p if curr_excite else 0
        self.tune_param = tune_param
        self.adapt_tune = adapt_tune

        # -> parameter about trace
        self.neu_tr = None
        self.tr_decay = tr_decay
        self.hard_mask = hard_mask

        # ->parameter about adaptive threshold
        self.adaptive_threshold = adaptive_threshold
        self.last_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)
        self.neu_act_cnts = torch.zeros((1, self.neu_size), requires_grad=False).to(device)
        self.task_num = 1

        if adaptive_threshold:
            self.thresh_max = thresh_max
            self.thresh_min = thresh_min
            self.adap_param = adap_param
            self.adap_approach = adap_approach
        self.v_threshold = torch.full((neu_size,), self.v_threshold).to(device)
        # -> parameter about neuron itself
        self.k_min = k_min

        if self.neu_type == 'if':
            self.charge_func = self.if_neuron_charge
        elif self.neu_type == 'lif':
            self.charge_func = self.lif_neuron_charge
        else:
            raise NotImplementedError("only have charge function")
        #################################
        # generate a name for the model #
        #################################
        inh_label = "{excite}{inhibit}{scale}".format(
            inhibit="" if not self.last_inhibit else "-inh",
            excite="" if not self.curr_excite else f"{'-aexc' if self.adapt_tune else '-exc'}",
            scale=f"s{self.tune_param}" if self.last_inhibit or self.curr_excite else "",
        )
        tr_label = "-{tr_decay}".format(
            tr_decay=f"d{self.tr_decay}",
        )
        neu_label = "{neu_type}{threshold}".format(
            neu_type=self.neu_type,
            threshold="" if not self.adaptive_threshold
            else "-{approach}{scale}{parameter}".format(
                approach="l" if self.adap_approach == "linear" else "ex",
                scale=f"{self.thresh_min}_{self.thresh_max}",
                parameter=f"_p{int(self.adap_param)}",
            )
        )
        k_label = "-{approach}{parameters}".format(
            approach="hard_" if self.hard_mask else "soft_",
            parameters=f"k{self.k_min}",
        )
        self.label = f"-kspktr{neu_label}{k_label}{tr_label}{inh_label}"

    @property
    def name(self):
        return self.label

    def if_neuron_charge(self, x: torch.Tensor):
        self.v = self.v + x

    def lif_neuron_charge(self, x: torch.Tensor):
        # 如果想要使用lif-neuron的话就必须将阈值上限设置为1，但是刚开始写的时候并没有注意，
        # 所以现在没法观察了（不过估计神经元类型也不会产生太大影响就是了
        # 另外一方面原因在于对第一层的输入进行了归一化导致输入值普遍比较小...
        self.v = self.v + (x - self.v) / self.tau
        # self.v = self.v * (1. - 1. / self.tau) + x

    def get_curr_k(self, curr_ep=None):
        k = self.k_min
        return k

    def top_k_fire(self, curr_ep=None, training=True, final_step=False):
        curr_k = self.get_curr_k(curr_ep=curr_ep)
        # -> reset the neu_tr
        if self.neu_tr is None:
            self.neu_tr = torch.zeros_like(self.v).to(self.v.data.device)

        spike = self.surrogate_function(self.v - self.v_threshold)
        self.neu_tr = self.neu_tr - self.neu_tr / self.tr_decay + spike.detach()

        if training:
            self.adap_neu_tr = self.spike_sigmoid(self.neu_tr)
        else:
            self.adap_neu_tr = self.neu_tr.clone()

        # -> implement the WTA-k to the spike
        if self.hard_mask or not final_step:
            vals, inds = torch.topk(self.neu_tr, curr_k + 1, dim=-1, sorted=False)
            thresh, _ = torch.min(vals.detach(), dim=-1, keepdim=True)  # -> get the no.k's threshold

            top_k_mask = (self.neu_tr > thresh).float()
        else:  # pure soft mask is not usable for CL, but we can only activate the last step's spike to at least make the learning convergence

            rand_ind = torch.randperm(self.neu_tr.shape[-1]).to(self.v.device)
            vals, inds = torch.topk(self.neu_tr[..., rand_ind], curr_k, dim=-1, sorted=False)
            thres, _ = torch.min(vals.detach(), dim=-1, keepdim=True)

            top_k_mask = torch.zeros_like(self.neu_tr)
            top_k_mask = top_k_mask.scatter(-1, rand_ind[inds], 1)

        spike_out = top_k_mask * spike

        # -> reset the mem of the neurons
        spike_d = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_d * self.v_threshold
        else:
            self.v = (1. - spike_d) * self.v + spike * self.v_reset

        # -> implement the adaptive threshold
        if training:
            self.neu_act_cnts += torch.sum(spike_out, dim=0, keepdim=True).detach()
        if self.adaptive_threshold:
            if self.adap_approach == 'linear':
                raw_thresh = self.thresh_min + (self.neu_act_cnts + self.last_act_cnts) / self.adap_param
            elif self.adap_approach == "exponential":
                raw_thresh = self.thresh_min + (self.thresh_max - self.thresh_min) * (
                        1. - torch.exp(-(self.neu_act_cnts + self.last_act_cnts) * 2 / self.adap_param))
            else:
                raise NotImplementedError("only support <linear> and <exponential> for adaptive threshold")
            self.v_threshold = torch.minimum(
                torch.ones_like(self.v_threshold) * self.thresh_max,
                torch.maximum(
                    torch.ones_like(self.v_threshold) * self.thresh_min,
                    raw_thresh,
                )
            )

        return spike_out

    def single_step_forward(self, x: torch.Tensor, curr_ep=None, training=True, final_step=False, **kwargs):
        assert self.step_mode == 'm', 'This topk node only support multi-step for convenience'
        self.v_float_to_tensor(x)
        self.charge_func(x)
        spike = self.top_k_fire(curr_ep=curr_ep, training=training, final_step=final_step)
        # 注意还要考虑一下每次初始化的问题
        # self.stdp_func(spike.detach())
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor, curr_ep=None, training=True, **kwargs):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            final_step = True if (t == T - 1) and torch.sum(torch.stack(y_seq)) < self.k_min else False
            y = self.single_step_forward(x_seq[t], training=training, final_step=final_step)
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        self.neu_tr = None

        return torch.stack(y_seq)

    def spike_sigmoid(self, sum_spike):
        summed_param = self.neu_act_cnts * self.excite_p - self.last_act_cnts * self.inhibit_p
        return sum_spike * (torch.sigmoid(summed_param) * self.tune_param + (1. - self.tune_param/2))

    def end_one_task(self):
        self.last_act_cnts += self.neu_act_cnts
        self.neu_act_cnts = torch.zeros_like(self.neu_act_cnts)
        if self.adapt_tune:
            self.excite_p = self.excite_p * (self.task_num + 1) / self.task_num
            self.task_num += 1

#######################################################################
# 也许可以用得上的一些科技？
# 这个东西的想法其实也很简单，由于脉冲的值都是完全一致的，所以可以通过最直接地随机筛选来限制输出频率（当然这也是有利有弊的做法就是了）
class Spike_Filter(nn.Module):
    def __init__(self, filter_prop):
        super(Spike_Filter, self).__init__()
        self.filter_prop = filter_prop

    def forward(self, x: torch.Tensor, **kwargs):
        # # 这边必须默认x是脉冲输入，不然这个东西就没法做
        # # 感觉同一batch不同样本之间搞混似乎不太好，但是如果考虑到这种问题的话似乎就会失去batch train的优势
        # # 嘶...
        # mask = torch.zeros_like(x)
        # neu_size = x.nelement()
        # max_freq = int(self.filter_prop * neu_size)
        # spk_sum = int(x.sum().item())
        # passed_num = min(max_freq, spk_sum)
        # # 这个是随机取样按固定数字取样
        # # spk_ind = torch.nonzero(x, as_tuple=True)
        # # choosed_gp = np.random.choice(list(range(len(spk_ind[0]))), passed_num, replace=False)
        # # mask[tuple([tup[choosed_gp] for tup in spk_ind])] = 1
        # # 这个是直接按照概率取
        # pass_prop = passed_num / spk_sum
        # prop = torch.rand_like(x)
        # mask = torch.where((x > 0) & (prop < pass_prop), x, mask).detach()

        # 上面那个部分是对整个batch进行概率过滤的写法，但是想想感觉很不对劲，考虑到输出脉冲规模未必均衡还是按照每条数据来吧
        mask = torch.zeros_like(x)

        max_freq = x.shape[-1] * self.filter_prop
        pass_prop = max_freq / torch.sum(x, dim=-1, keepdim=True)
        prop = torch.rand_like(x)

        mask[prop < pass_prop] = 1

        return mask * x


#######################################################################
# several neurons which can have individual threshold for each neuron
class Adapt_Thresh_IFNode(neuron.IFNode):
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            self.v = self.v - spike_d * self.v_threshold
        else:
            self.v = (1 - spike_d) * self.v + spike_d * self.v_reset

    def single_step_forward(self, x: torch.Tensor):
        return super(neuron.IFNode, self).single_step_forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor):
        return super(neuron.IFNode, self).multi_step_forward(x_seq)


class Adapt_Thresh_LIFNode(neuron.LIFNode):
    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            self.v = self.v - spike_d * self.v_threshold
        else:
            self.v = (1 - spike_d) * self.v + spike_d * self.v_reset

    def single_step_forward(self, x: torch.Tensor):
        return super(neuron.LIFNode, self).single_step_forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor):
        return super(neuron.LIFNode, self).multi_step_forward(x_seq)


