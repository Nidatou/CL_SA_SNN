from torch import nn
import torch
import numpy as np
from models.utils import utils
from models.fc.layers import fc_layer, fc_layer_fixed_gates, fc_layer_split, fc_attention
from models.fc.layers import Top_K

from spikingjelly.activation_based import neuron, surrogate


#######################################################
# a formal MLP layer to deal with the features
class MLP(nn.Module):
    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, size_per_layer=None, drop=0.,
                 neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid(), gate_buffer=True, output='normal'):
        assert layers > 0
        super().__init__()
        self.output = output
        # get sizes of all layers
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size]
        self.layers = len(size_per_layer) - 1

        # -set label ######################################
        # set labels for MLP
        nd_label = "{drop}{nl}".format(
            drop="" if drop == 0. else f"d{drop}",
            nl="IF" if neuron_type == neuron.IFNode else "LIF"
        )
        nd_label = "{}{}".format(nd_label, "" if output == "normal" else f"-{output}")

        size_statement = ""
        for i in size_per_layer:
            size_statement += "{}{}".format("-" if size_statement == "" else "x", i)
        self.label = f"F{size_statement}{nd_label}" if self.layers > 0 else ""

        # set layers
        for layer_id in range(1, self.layers+1):
            # get the number of the layer's input and output
            in_size = size_per_layer[layer_id-1]
            out_size = size_per_layer[layer_id]
            layer = fc_layer(
                in_size, out_size, gate_buffer=gate_buffer, drop=drop, surr_func=surr_func,
                neuron_type=utils.Identity if layer_id == self.layers else neuron_type
            )
            setattr(self, f'fcLayer{layer_id}', layer)
        if output == "normal":
            self.final_neu = neuron_type(surrogate_function=surr_func)
        elif output == 'sigmoid':
            self.final_neu = nn.Sigmoid()
        else:
            self.final_neu = utils.Identity()

    def forward(self, x, **kwargs):
        for layer_id in range(1, self.layers + 1):
            x = getattr(self, f"fcLayer{layer_id}")(x)
        x = self.final_neu(x)
        return x

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        lyr_list = []
        for layer_id in range(1, self.layers+1):
            lyr_list += getattr(self, f"fcLayer{layer_id}").list_init_layers()
        return lyr_list


#######################################################
# the gated MLP used for VAE decoders
class MLP_gates(nn.Module):
    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, size_per_layer=None, drop=0.,
                 neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid(), gate_buffer=True, output='normal',
                 gate_size=0, gating_prop=0., final_gate=False, device="cuda:0"):
        assert layers > 0
        super(MLP_gates, self).__init__()
        self.output = output

        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                hidden_sizes = [int(x) for x in np.repeat(hid_size, layers-1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size]
        self.layers = len(size_per_layer) - 1

        # -set label ######################################
        # set labels for MLP, gate_size devote the size of input task-signal
        nd_label = "{drop}{nl}{gate}".format(
            drop="" if drop == 0. else f"d{drop}",
            nl="IF" if neuron_type == neuron.IFNode else "LIF",
            gate="g{}m{}".format(gate_size, gating_prop) if (gate_size > 0 and gating_prop > 0.) else ""
        )
        nd_label = "{}{}".format(nd_label, "" if output == "normal" else f"-{output}")

        size_statement = ""
        for i in size_per_layer:
            size_statement += "{}{}".format("-" if size_statement == "" else "x", i)
        self.label = f"F{size_statement}{nd_label}" if self.layers > 0 else ""

        for layer_id in range(1, self.layers+1):
            in_size = size_per_layer[layer_id-1]
            out_size = size_per_layer[layer_id]
            if gate_size <= 0. or gating_prop <= 0. or (layer_id == self.layers and not final_gate):
                layer = fc_layer(
                    in_size, out_size, gate_buffer=gate_buffer, surr_func=surr_func, drop=drop,
                    neuron_type=utils.Identity if ((output == "none" or output == 'sigmoid') and layer_id == self.layers) else neuron_type
                )
            else:
                layer = fc_layer_fixed_gates(
                    in_size, out_size, gate_buffer=gate_buffer, surr_func=surr_func, drop=drop,
                    neuron_type=utils.Identity if ((output == "none" or output == 'sigmoid') and layer_id == self.layers) else neuron_type,
                    gate_size=gate_size, gating_prop=gating_prop, device=device,
                )
            setattr(self, f"fcLayer{layer_id}", layer)
        self.final_neu = utils.Identity() if output != 'sigmoid' else nn.Sigmoid()

    def forward(self, x, gate_input=None):
        for layer_id in range(1, self.layers+1):
            x = getattr(self, f"fcLayer{layer_id}")(x, gate_input=gate_input)
        x = self.final_neu(x)
        return x

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        lyr_list = []
        for layer_id in range(1, self.layers+1):
            lyr_list += getattr(self, f"fcLayer{layer_id}").list_init_layers()
        return lyr_list


#######################################################
# the gated MLP with feedforward gate
class MLP_attention(nn.Module):
    def __init__(self, input_size=1000, output_size=10, layers=2, hid_size=1000, size_per_layer=None, drop=0.,
                 neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid(), gate_buffer=True, output='normal',
                 atten_input_size=None, attention_type="normal"):
        assert layers > 0
        assert attention_type in ["normal", "recurrent"]
        super(MLP_attention, self).__init__()
        self.output = output
        if size_per_layer is None:
            hidden_sizes = []
            if layers > 1:
                hidden_sizes = [int(x) for x in np.repeat(hid_size, layers - 1)]
            size_per_layer = [input_size] + hidden_sizes + [output_size]
        self.layers = len(size_per_layer) - 1

        # -set label ######################################
        # set labels for MLP
        nd_label = "{drop}{nl}".format(
            drop="" if drop == 0. else f"d{drop}",
            nl="IF" if neuron_type == neuron.IFNode else "LIF"
        )
        nd_label = "{}{}".format(nd_label, "" if output == "normal" else f"-{output}")
        nd_label += "-attention{}".format("-R" if attention_type == "recurrent" else "")

        size_statement = ""
        for i in size_per_layer:
            size_statement += "{}{}".format("-" if size_statement == "" else "x", i)
        self.label = f"F{size_statement}{nd_label}" if self.layers > 0 else ""

        if atten_input_size is None:
            atten_input_size = input_size

        for layer_id in range(1, self.layers + 1):
            in_size = size_per_layer[layer_id - 1]
            out_size = size_per_layer[layer_id]
            layer = fc_layer(
                in_size, out_size, gate_buffer=gate_buffer, drop=drop, surr_func=surr_func,
                neuron_type=utils.Identity if (
                            (output == "none" or output == 'sigmoid') and layer_id == self.layers) else neuron_type
            )
            setattr(self, f'fcLayer{layer_id}', layer)

            atten_layer = fc_attention(
                atten_input_size, out_size, drop=drop, surr_func=surr_func,
                neuron_type=neuron_type, forward_method=attention_type,
            )
            setattr(self, f'AttLayer{layer_id}', atten_layer)
        self.atten_record = []
        self.final_neu = utils.Identity() if output == 'none' else neuron_type(surrogate_function=surr_func)

    def forward(self, x: torch.Tensor, attention_input=None, record_atten=False, **kwargs):
        if attention_input is None:
            attention_input = x
        self.atten_record = []
        for layer_id in range(1, self.layers+1):
            x = getattr(self, f"fcLayer{layer_id}")(x)  # -> (T, N, H)
            atten = getattr(self, f"AttLayer{layer_id}")(attention_input)  # -> (T, N, H)
            x = x * atten
            if record_atten:
                # record the gate results
                self.atten_record.append(np.mean(np.mean(atten.cpu().numpy(), axis=0), axis=0))
        x = self.final_neu(x)
        return x

    def freeze_atten(self):
        for layer_id in range(1, self.layers+1):
            for param in getattr(self, f"AttLayer{layer_id}").parameters():
                param.requires_grad = False

    @property
    def name(self):
        return self.label

    def list_init_layers(self):
        lyr_list = []
        for layer_id in range(1, self.layers+1):
            lyr_list += getattr(self, f"fcLayer{layer_id}").list_init_layers()
        for layer_id in range(1, self.layers+1):
            lyr_list += getattr(self, f"AttLayer{layer_id}").list_init_layers()
        return lyr_list


#######################################################
# Added Part for SDM
class SDM_Base(nn.Module):
    def __init__(
        self, input_size, nneu, output_size, device=None,
        # -> Top-K parameter
        k_mask=False, k_approach="LINEAR_DACAY", k_min=10, k_max=None,
        k_trans_ep=None, gaba_switch_num=None,
        # -> parameter_regularisation
        norm_ad=True, norm_val=False, dale=True,
    ):
        super(SDM_Base, self).__init__()
        self.input_size = input_size
        self.nneurons = nneu

        self.fc1 = nn.Linear(input_size, nneu, bias=False)
        k_max = nneu if k_max is None else k_max
        self.top_k = Top_K(nneu=nneu, k_approch=k_approach, k_min=k_min, k_max=k_max, device=device,
                           k_trans_ep=k_trans_ep, gaba_switch_num=gaba_switch_num, mask=k_mask)
        self.purkinje_layer = nn.Linear(nneu, output_size, bias=False)

        self.norm_ad = norm_ad
        self.norm_value = norm_val
        self.dale = dale

        topk_label = "(topk_{kmin}-{kmax}_{approach_tag}-{hyper}{mask})".format(
            kmin=k_min, kmax=k_max, hyper=gaba_switch_num if "GABA_SWITCH" in k_approach else k_trans_ep,
            approach_tag="GABA" if "GABA_SWITCH" in k_approach else ("LINEAR" if "LINEAR" in k_approach else "FLAT"),
            mask="-M" if k_mask else "",
        )
        regu_label = '{}{}'.format("-Na" if norm_ad else "", "-Nval" if norm_val else "",)
        regu_label = '{}{}'.format(regu_label, '-D' if self.dale else '')
        self.label = f"-{input_size}x{nneu}{topk_label}x{output_size}{regu_label}"

    @property
    def name(self):
        return self.label

    def enforce_weight_regularization(self):
        if self.dale:
            with torch.no_grad():
                # self.fc1.weight.data.clamp_(0)
                # self.purkinje_layer.weight.data.clamp_(0)
                for p in list(self.parameters()):
                    p.data.clamp_(0)

        if self.norm_ad:
            with torch.no_grad():
                self.fc1.weight.data /= torch.norm(self.fc1.weight.data, dim=1, keepdim=True)
        if self.norm_value:
            with torch.no_grad():
                self.purkinje_layer.weight.data /= torch.norm(self.purkinje_layer.weight.data, dim=1, keepdim=True)

    def forward(self, x: torch.Tensor, cur_ep=None, training=True):
        # if self.dale:
        #     x = nn.ReLU()(x)
        if self.norm_ad:
            x = x / torch.norm(x, dim=1, keepdim=True)

        x = self.fc1(x)
        # => implement WTA-K, and need the ep to calculate the switch process
        x = self.top_k(x, cur_ep=cur_ep, training=training)
        active_out = torch.clone(x.detach())
        x = self.purkinje_layer(x)

        return x

    def mid_forward(self, x: torch.Tensor, cur_ep, **kwargs):
        if self.dale:
            x = nn.ReLU()(x)
        if self.norm_ad:
            x = x / torch.norm(x, dim=1, keepdim=True)

        x = self.fc1(x)
        # => implement WTA-K, and need the ep to calculate the switch process
        x = self.top_k(x, cur_ep=cur_ep, training=False)
        return x


class Three_Layer_MLP(nn.Module):
    def __init__(
        self, input_size, nneu, output_size, device=None,
        norm_ad=True, norm_val=False, dale=True,
    ):
        super(Three_Layer_MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, nneu)
        self.output_layer = nn.Linear(nneu, output_size)
        self.nl = nn.ReLU()

        self.norm_ad = norm_ad
        self.norm_value = norm_val
        self.dale = dale

        added_tag = "{nd}{nv}{d}".format(
            nd="-nad" if norm_ad else '', nv='-nv' if norm_val else '', d="-d" if dale else '',
        )
        self.label = f"-simple_mlp{input_size}x{nneu}x{output_size}{added_tag}"

    def enforce_weight_regularization(self):
        if self.dale:
            with torch.no_grad():
                # self.fc1.weight.data.clamp_(0)
                # self.purkinje_layer.weight.data.clamp_(0)
                for p in list(self.parameters()):
                    p.data.clamp_(0)

        if self.norm_ad:
            with torch.no_grad():
                self.fc1.weight.data /= torch.norm(self.fc1.weight.data, dim=1, keepdim=True)
        if self.norm_value:
            with torch.no_grad():
                self.purkinje_layer.weight.data /= torch.norm(self.purkinje_layer.weight.data, dim=1, keepdim=True)

    @property
    def name(self):
        return self.label

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.fc1(x)
        x = self.nl(x)
        x = self.output_layer(x)
        return x

    def mid_forward(self, x: torch.Tensor, cur_ep, **kwargs):
        if self.dale:
            x = nn.ReLU()(x)
        if self.norm_ad:
            x = x / torch.norm(x, dim=1, keepdim=True)
        x = self.fc1(x)
        x = self.nl(x)
        return x

