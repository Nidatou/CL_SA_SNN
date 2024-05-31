from torch import nn
import torch
import numpy as np

from spikingjelly.activation_based import neuron, layer, surrogate, functional

import models.conv.layers as conv_layers
from models.utils import utils


class SNN_ConvLayer(nn.Module):
    """Convolutional feature extractor model for (natural) images. Possible to return (pre)activations of each layer.
    Also possible to supply a [skip_first]- or [skip_last]-argument to the forward-function to only pass certain layers.

    Input:  [T, B, C, W, H] tensor
    Output: [T, B, c, w, h] tensor
    """
    def __init__(self, conv_type="standard", block_type="basic", num_blocks=2,
                 image_channels=3, depth=5, start_channels=16, reducing_layers=None, batch_norm=True,
                 neuron_type=neuron.IFNode, surr_func=surrogate.Sigmoid(), output="normal", global_pooling=False):
        # 有必要再补上介绍吧
        assert conv_type in ['standard', 'resNet']
        assert block_type in ['bottleneck', 'basic']
        conv_type = "standard" if depth < 2 else conv_type
        if conv_type == 'resNet':
            num_blocks = [num_blocks]*(depth-1) if type(num_blocks) == int else num_blocks
            assert len(num_blocks) == (depth - 1)
            block = conv_layers.Bottleneck if block_type == 'bottleneck' else conv_layers.BasicBlock
        if reducing_layers is not None:
            reducing_layers = reducing_layers if depth >= reducing_layers else depth

        # Generate the label
        type_label = "C" if conv_type == 'standard' else f"R{'b' if block_type=='bottleneck' else''}"
        channel_label = f"{image_channels}-{depth}x{start_channels}"
        block_label = f"-{num_blocks}" if conv_type == 'resNet' else ""
        nd_label = "{bn}{gp}{out}".format(bn="b" if batch_norm else "",
                                          gp="p" if global_pooling else "",
                                          out="n" if output == "none" else "")
        nd_label = f"-{nd_label}" if nd_label != "" else nd_label

        # set configuration
        super(SNN_ConvLayer, self).__init__()
        self.depth = depth

        self.rl = depth - 1 if (reducing_layers is None) else reducing_layers
        rl_label = "" if self.rl == (self.depth-1) else f"-rl{self.rl}"
        self.label = f"{type_label}{channel_label}{block_label}{nd_label}{rl_label}"

        self.block_expansion = block.expansion if conv_type == "resNet" else 1
        # double factor is the index when stop expansion between blocks
        double_factor = self.rl if self.rl < depth else depth -1
        self.out_channels = (start_channels * 2**double_factor) * self.block_expansion if depth > 0 else image_channels
        self.start_channels = start_channels
        self.global_pooling = global_pooling

        # Conv-layers
        output_channels = start_channels
        for layer_id in range(1, depth+1):
            reducing = True if (layer_id > depth - self.rl) else False
            input_channels = image_channels if layer_id == 1 else output_channels * self.block_expansion
            output_channels = output_channels * 2 if (reducing and not layer_id == 1) else output_channels
            # sequentially define the convolutional-layer
            NFL = (output == "none" and layer_id == depth)
            if conv_type == 'standard' or layer_id == 1:
                conv_layer = conv_layers.conv_layer(input_channels, output_channels, stride=2 if reducing else 1,
                                                    drop=0, neuron_type=utils.Identity if NFL else neuron_type,
                                                    surr_func=surr_func, batch_norm=False if NFL else batch_norm)
            else:
                conv_layer = conv_layers.res_layer(input_channels, output_channels, block=block,
                                                   num_blocks=num_blocks[layer_id-2], stride=2 if reducing else 1,
                                                   drop=0, batch_norm=batch_norm, neuron_type=neuron_type,
                                                   surr_func=surr_func, no_neu=True if NFL else False)
            setattr(self, f"convLayer{layer_id}", conv_layer)

        self.pooling = layer.AdaptiveAvgPool2d((1, 1)) if global_pooling else utils.Identity()

    def forward(self, x: torch.Tensor, skip_first=0, skip_last=0):

        for layer_id in range(skip_first+1, self.depth+1-skip_last):
            x = getattr(self, f"convLayer{layer_id}")(x)
        # global average if requested
        x = self.pooling(x)
        return x

    # return the output size of the Conv-Layer area
    def out_size(self, image_size, ignore_gp=False):
        out_size = int(np.ceil(image_size / 2 ** self.rl)) if self.depth > 0 else image_size
        return 1 if (self.global_pooling and not ignore_gp) else out_size

    # return the output units of the Conv-Layer
    def out_units(self, image_size, ignore_gp=False):
        return self.out_channels * self.out_size(image_size, ignore_gp=ignore_gp) ** 2

    # return the layer info of all hidden layer
    def layer_info(self, image_size):
        layer_list = []
        reduce_number = 0  # -> record how often image-size has beem halved
        double_number = 0  # -> record how often channel number has been doubled
        for layer_id in range(1, self.depth):
            reducing = True if (layer_id > self.depth - self.rl) else False
            if reducing:
                reduce_number += 1
            if reducing and layer_id > 1:
                double_number += 1
            pooling = True if self.global_pooling and layer_id == self.depth-1 else False
            expansion = 1 if layer_id == 1 else self.block_expansion
            layer_list.append([(self.start_channels * 2**double_number)*expansion,
                               1 if pooling else int(np.ceil(image_size / 2**reduce_number)),
                               1 if pooling else int(np.ceil(image_size / 2**reduce_number))])
        return layer_list

    # return the layer list of all the hidden block (about their parameters)
    def list_init_layers(self):
        lyr_list = []
        for layer_id in range(1, self.depth+1):
            lyr_list += getattr(self, f"convLayer{layer_id}").list_init_layers()
        return lyr_list

    @property
    def name(self):
        return self.label


class VGGSNNwoAP(nn.Module):
    def __init__(
        self, image_channels=3, batch_norm=True, neu_type=neuron.IFNode, surr_func=surrogate.Sigmoid(),
    ):
        super(VGGSNNwoAP, self).__init__()
        self.conv_list = [
            [64, 3, 1],
            [128, 3, 2],
            [256, 3, 1],
            [256, 3, 2],
            [512, 3, 1],
            [512, 3, 2],
            [512, 3, 1],
            [512, 3, 2],
        ]
        type_label = "VGGwoAP"
        channel_label = f"{image_channels}-{len(self.conv_list)}x{self.conv_list[0][0]}"
        other_label = "-bn" if batch_norm else ""
        self.label = f"{type_label}_{channel_label}{other_label}"

        self.depth = len(self.conv_list)
        self.neu_type = neu_type
        self.surr_func = surr_func
        self.block_expansion = 1
        self.out_channels = self.conv_list[-1][0]
        input_channels = image_channels
        for layer_id in range(len(self.conv_list)):
            output_channels = self.conv_list[layer_id][0]
            conv_layer = conv_layers.conv_layer(
                input_channels, output_channels, stride=self.conv_list[layer_id][2],
                kernel_size=self.conv_list[layer_id][1], drop=0, neuron_type=self.neu_type, surr_func=surr_func,
                batch_norm=batch_norm
            )
            setattr(self, f"convLayer{layer_id+1}", conv_layer)
            input_channels = output_channels * self.block_expansion

    def forward(self, x: torch.Tensor, **kwargs):
        for layer_id in range(1, self.depth+1):
            x = getattr(self, f"convLayer{layer_id}")(x)
        return x

    # return the output size of the Conv-Layer area
    def out_size(self, image_size, ignore_gp=False):
        out_size = int(np.ceil(image_size / 2 ** 4))
        return out_size

    # return the output units of the Conv-Layer
    def out_units(self, image_size, ignore_gp=True):
        out_units = self.out_channels * self.out_size(image_size, ignore_gp=ignore_gp) ** 2
        print(f"{image_size}-{self.out_size(image_size, ignore_gp=ignore_gp)}")
        return out_units

    # return the layer info of all hidden layer
    def layer_info(self, image_size):
        return self.conv_list

    # return the layer list of all the hidden block (about their parameters)
    def list_init_layers(self):
        lyr_list = []
        for layer_id in range(1, self.depth+1):
            lyr_list += getattr(self, f"convLayer{layer_id}").list_init_layers()
        return lyr_list

    @property
    def name(self):
        return self.label


class Res_conv(nn.Module):
    def __init__(
        self, block, layers, zero_init_residual=False,
        groups=1, width_per_group=64, replace_stride_with_dilation=None,
        conv_neu=neuron.LIFNode, surr_func=surrogate.ATan(), T=8
    ):
        super(Res_conv, self).__init__()

        self.inplanes = 64
        self.dilation = 1
        self.conv_neu = conv_neu
        self.surr_func = surr_func
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_s = nn.Sequential(
            layer.Conv2d(in_channels=3, out_channels=self.inplanes, kernel_size=3, stride=1, padding=1,
                         bias=False),
            layer.BatchNorm2d(self.inplanes)
        )
        self.neu_v1 = conv_neu(surrogate_function=surr_func)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, conv_layers.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        model_label = f"R3-{len(layers) + 1}{width_per_group}-resNet"
        nd_label = "-bp"
        self.label = f"{model_label}{nd_label}"

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                            conv_neu=self.conv_neu, surr_func=self.surr_func))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation,
                      conv_neu=self.conv_neu, surr_func=self.surr_func))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.neu_v1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        return x

    @property
    def name(self):
        return self.label
