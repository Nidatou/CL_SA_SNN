import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, layer, surrogate, functional

from models.utils import utils


##################################################
# ResNet-blocks ############
# (2 conv with expansion=1)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, batch_norm=True,
                 neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid(), no_neu=False,  **kwargs):
        super(BasicBlock, self).__init__()
        self.n_t = neuron_type
        self.surr = surr_func

        # self.grad_with_rate = kwargs.get('grad_with_rate', False)
        # normal block-layers
        self.block_layer1 = nn.Sequential(
            layer.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            layer.BatchNorm2d(planes) if batch_norm else utils.Identity(),
            self.n_t(surrogate_function=self.surr),
        )
        self.block_layer2 = nn.Sequential(
            layer.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(self.expansion*planes) if batch_norm else utils.Identity(),
        )

        # shortcut block-layer
        self.shortcut = utils.Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            )

        self.final_neuron = self.n_t(surrogate_function=self.surr) if not no_neu else utils.Identity()

    def forward(self, x: torch.Tensor):
        out = self.block_layer2(self.block_layer1(x))
        out += self.shortcut(x)
        final_opt = self.final_neuron(out)
        return final_opt

    def list_init_layers(self):
        lyr_list = [self.block_layer1[0], self.block_layer2[0]]
        if not type(self.shortcut) == utils.Identity:
            lyr_list.append(self.shortcut)
        return lyr_list


class Spk_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 conv_neu=neuron.LIFNode, surr_func=surrogate.ATan()):
        super(Spk_BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1_s = nn.Sequential(
            layer.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride,
                         padding=1, groups=1, bias=False, dilation=1),
            layer.BatchNorm2d(planes)
        )
        self.neu_1 = conv_neu(surrogate_function=surr_func)
        self.conv2_s = nn.Sequential(
            layer.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1,
                         padding=1, groups=1, bias=False, dilation=1),
            layer.BatchNorm2d(planes)
        )
        self.neu_2 = conv_neu(surrogate_function=surr_func)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_s(x)
        out = self.neu_1(out)
        out = self.conv2_s(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.neu_2(out)

        return out


# building block with bottleneck for ResNets (3 conv with expansion=4)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, batch_norm=True,
                 neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid(), no_neu=False, **kwargs):
        super(Bottleneck, self).__init__()
        self.n_t = neuron_type
        self.surr = surr_func
        self.block_layer1 = nn.Sequential(
            layer.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            layer.BatchNorm2d(planes) if batch_norm else utils.Identity(),
            self.n_t(surrogate_function=self.surr),
        )
        self.block_layer2 = nn.Sequential(
            layer.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            layer.BatchNorm2d(planes) if batch_norm else utils.Identity(),
            self.n_t(surrogate_function=self.surr),
        )
        self.block_layer3 = nn.Sequential(
            layer.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False),
            layer.BatchNorm2d(self.expansion*planes) if batch_norm else utils.Identity(),
        )

        # shortcut block-layer
        self.shortcut = utils.Identity()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                layer.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=1, bias=False),
                layer.BatchNorm2d(self.expansion*planes) if batch_norm else utils.Identity(),
            )

        # final neuron layers
        self.final_neuron = self.n_t(surrogate_function=self.surr) if not no_neu else utils.Identity()

    def forward(self, x):
        out = self.block_layer3(self.block_layer2(self.block_layer1(x)))
        out += self.shortcut(x)
        final_out = self.final_neuron(out)
        return final_out

    def list_init_layers(self):
        lyr_list = [self.block_layer1[0], self.block_layer2[0], self.block_layer3[0]]
        if not type(self.shortcut) == utils.Identity:
            lyr_list.append(self.shortcut)
        return lyr_list


##################################################
# Conv-layers ############
# I will reduce the department which seems needless in the whole method
class conv_layer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1,
                 drop=0, batch_norm=False, neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid()):
        super(conv_layer, self).__init__()
        if drop > 0:
            self.dropout = layer.Dropout2d(drop)
        self.conv = layer.Conv2d(in_planes, out_planes, stride=stride, kernel_size=kernel_size, padding=padding, bias=False)
        if batch_norm:
            self.bn = layer.BatchNorm2d(out_planes)
        self.final_neuron = neuron_type(surrogate_function=surr_func)

    def forward(self, x: torch.Tensor):
        input = self.dropout(x) if hasattr(self, 'dropout') else x
        mem = self.conv(input) if not hasattr(self, 'bn') else self.bn(self.conv(input))
        output = self.final_neuron(mem)
        return output

    def list_init_layers(self):
        return [self.conv]


# convolution layer output [mean] and [logvar]
class conv_layer_split(nn.Module):
    def __init__(self, in_planes, out_planes, neu_mean=utils.NonSpikingIFNode, neu_logvar=utils.NonSpikingIFNode,
                 kernel_size=3, stride=1, padding=1, drop=0, batch_norm=False):
        super(conv_layer_split, self).__init__()
        self.mean = conv_layer(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               drop=drop, batch_norm=batch_norm, neuron_type=neu_mean)
        self.logvar = conv_layer(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                 drop=drop, batch_norm=batch_norm, neuron_type=neu_logvar)

    def forward(self, x: torch.Tensor):
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

    def list_init_layers(self):
        list = []
        list += self.mean.list_init_layers()
        list += self.logvar.list_init_layers()
        return list


# snn convolution res-net layer
class res_layer(nn.Module):
    def __init__(self, in_planes, out_planes, block=BasicBlock, num_blocks=2, stride=1, drop=0, batch_norm=True,
                neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid(), no_neu=False):

        super(res_layer, self).__init__()
        self.num_blocks = num_blocks
        self.in_planes = in_planes
        self.out_planes = out_planes * block.expansion

        self.dropout = layer.Dropout2d(drop)
        for block_id in range(num_blocks):
            new_block = block(in_planes, out_planes, stride=stride if block_id == 0 else 1, batch_norm=batch_norm,
                              neuron_type=neuron_type, surr_func=surr_func,
                              no_neu=True if block_id == (num_blocks-1) else False)
            setattr(self, f"block{block_id+1}", new_block)
            in_planes = out_planes * block.expansion

        self.final_neuron = neuron_type(surrogate_function=surr_func) if not no_neu else utils.Identity()

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        for block_id in range(self.num_blocks):
            x = getattr(self, f"block{block_id+1}")(x)
        output = self.final_neuron(x)
        return output

    def list_init_layers(self):
        list = []
        for block_id in range(self.num_blocks):
            list += getattr(self, f"block{block_id+1}").list_init_layers()
        return list


