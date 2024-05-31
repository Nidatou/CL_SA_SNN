import numpy as np
import torch
import torch.nn.functional as F

from models.utils import utils, loss_func
from models.cl.continual_learner import CLBase
from models.conv.nets import SNN_ConvLayer, VGGSNNwoAP
from models.fc.nets import MLP
from models.fc.layers import fc_layer
from models.utils.loss_func import TET_loss

from spikingjelly.activation_based import neuron, surrogate, layer, functional, encoding


class SNN_Classifier(CLBase):
    def __init__(
        self, image_size, image_channels, classes, T,
        # conv-layers-part
        conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True,
        conv_neu=neuron.IFNode, num_blocks=2, global_pooling=False, no_neu=True,
        # fc-layers-part
        fc_depth=3, fc_units=1000, h_dim=400, fc_drop=0., fc_neu=neuron.IFNode, gate_buffer=True,
        # other setting
        hidden=False, surr_func=surrogate.Sigmoid(), final_neu=utils.Identity,
        loss_type='ce', loss_lambda=1e-3, loss_means=1.
    ):
        super(SNN_Classifier, self).__init__()
        self.label = "Classifier"
        self.classes = classes

        self.depth = depth
        self.fc_depth = fc_depth
        self.fc_drop = fc_drop

        self.hidden = hidden

        if self.fc_depth < 1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        ########################################################
        # model construction part ##############################
        # Conv Layers #############
        if conv_type == 'vggsnn':
            self.convE = VGGSNNwoAP(
                image_channels=image_channels, batch_norm=conv_bn, neu_type=conv_neu, surr_func=surr_func,
            )
        else:
            self.convE = SNN_ConvLayer(
                conv_type=conv_type, block_type="basic", num_blocks=num_blocks, image_channels=image_channels,
                depth=depth, start_channels=start_channels, reducing_layers=reducing_layers, batch_norm=conv_bn,
                neuron_type=conv_neu, surr_func=surr_func, global_pooling=global_pooling,
                output="none" if no_neu else "normal",
            )
        # 上面那个output估计得改掉
        self.flatten = layer.Flatten()
        # calculate output size #######
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels
        if fc_depth < 2:
            self.fc_layer_sizes = [self.conv_out_units]
        elif fc_depth == 2:
            self.fc_layer_sizes = [self.conv_out_units, h_dim]
        else:
            self.fc_layer_sizes = [self.conv_out_units] + [int(x) for x in np.linspace(fc_units, h_dim, num=fc_depth-1)]
        self.units_before_classifier = h_dim if fc_depth > 1 else self.conv_out_units

        # FC Layers ##################
        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, neuron_type=fc_neu, surr_func=surr_func,
                       gate_buffer=gate_buffer, output='normal')
        self.final_neuron = final_neu
        self.classifier = fc_layer(self.units_before_classifier, classes, gate_buffer=True, drop=0.,
                                   neuron_type=self.final_neuron, surr_func=surr_func)

        # --> SNN Setting ------------ ###
        self.T = T
        self.encoder = encoding.PoissonEncoder()
        assert loss_type in ['ce', 'tet']
        self.loss_type = loss_type
        self.loss_lamb = loss_lambda
        self.loss_means = loss_means
        if self.loss_type == 'tet':
            self.logits = None
        functional.set_step_mode(self, step_mode="m")

    def list_init_layers(self):
        list = self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        if self.depth > 0 and self.fc_depth > 1:
            return "{}_{}_c{}".format(self.convE.name, self.fcE.name, self.classes)
        elif self.depth > 0:
            return "{}_c{}".format(self.convE.name, self.classes)

        elif self.fc_depth > 1:
            return "{}_c{}".format(self.fcE.name, self.classes)
        else:
            return "i{}_c{}".format(self.fc_layer_sizes[0], self.classes)

    @property
    def classifier_weight(self):
        return self.classifier.linear.weight.data.detach().cpu().numpy()

    def forward(self, x: torch.Tensor, training=True, **kwargs):
        if len(x.shape) == 4 or len(x.shape) == 2:
            x = x.repeat(self.T, *(len(x.shape) * [1]))
        # x = self.encoder(x)
        hidden_rep = self.convE(x)
        final_features = self.fcE(self.flatten(hidden_rep))
        output = self.classifier(final_features)
        if self.loss_type == 'tet':
            self.logits = output
        output = output[-1] if self.final_neuron == utils.NonSpikingIFNode else torch.mean(output, 0)

        functional.reset_net(self)
        return output

    def check_neuron_activation(self, x: torch.Tensor):
        with torch.no_grad():
            if len(x.shape) == 4 or len(x.shape) == 2:
                x = x.repeat(self.T, *(len(x.shape) * [1]))
            hidden_rep = self.convE(x)
            final_features = self.fcE(self.flatten(hidden_rep))

            functional.reset_net(self)

        return torch.sum(final_features, dim=0).detach()

    def input_to_hidden(self, x: torch.Tensor):
        if len(x.shape) == 4 or len(x.shape) == 2:
            x = x.repeat(self.T, *(len(x.shape) * [1]))
        hidden_f = self.convE(x)
        functional.reset_net(self)
        return hidden_f

    def hidden_to_output(self, hidden_rep):
        output = self.classifier(self.fcE(self.flatten(hidden_rep)))
        output = output[-1] if self.final_neuron == utils.NonSpikingIFNode else torch.mean(output, 0)

        functional.reset_net(self)
        return output

    def classify(self, x: torch.Tensor, not_hidden=True, return_spk=False):
        if not_hidden and (len(x.shape) == 4 or len(x.shape) == 2):
            x = x.repeat(self.T, *(len(x.shape) * [1]))
        image_features = self.flatten(x) if (self.hidden and not not_hidden) else self.flatten(self.convE(x))
        hE = self.fcE(image_features)
        output = self.classifier(hE)
        output = output[-1] if self.final_neuron == utils.NonSpikingIFNode else torch.mean(output, 0)

        functional.reset_net(self)
        if return_spk:
            return output, torch.sum(hE, dim=0).detach()
        return output

    def train_a_batch(
        self, optimizer: torch.optim.Optimizer, x, y=None, x_=None, y_=None, scores_=None, rnt=0.5,
        active_classes=None, task=1, replay_not_hidden=False, freeze_convE=False, scenario='class', **kwargs
    ):
        self.train()
        if freeze_convE:
            self.convE.eval()

        optimizer.zero_grad()

        ################################################
        # loss in current data #########################
        if x is not None:
            # 如果采用了XdG还得对权重进行调整
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            if len(x.shape) == 4 or len(x.shape) == 2:
                batch_size = x.shape[0]
            else:
                batch_size = x.shape[1]

            y_hat = self(x)
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
                predL = None if y is None else loss_func.weighted_average(predL, weights=None, dim=0)
            elif self.loss_type == 'tet':
                predL = None if y is None else TET_loss(outputs=self.logits, labels=y, means=self.loss_means, lamb=self.loss_lamb)
            else:
                raise NotImplementedError(f"loss type <{self.loss_type}> is not supported")

            # Weigh losses
            loss_cur = predL
            # Calculate training-accuracy
            accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / batch_size

            # 在采用了XdG情况下，反向传播需要在mask更换之前提前进行反向传播
            if (self.mask_dict is not None) and (x_ is not None):
                weighted_current_loss = rnt * loss_cur
                weighted_current_loss.backward()

        else:
            accuracy = predL = None

        #################################################
        # loss in replayed data #########################
        if x_ is not None:
            y_ = [y_] if (y_ is not None and type(y_) != list) else y_
            scores_ = [scores_] if (scores_ is not None and type(scores_) != list) else scores_
            if active_classes is not None:
                assert type(active_classes) == list and len(active_classes) != 0
                if type(active_classes[0]) == list:
                    active_classes = active_classes
                else:
                    active_classes = [active_classes]
            # active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            loss_replay = [torch.tensor(0., device=self._device())] * n_replays
            predL_re = [torch.tensor(0., device=self._device())] * n_replays
            distilL_re = [torch.tensor(0., device=self._device())] * n_replays

            for replay_id in range(n_replays):
                if type(x_) == list or (
                        self.mask_dict is not None):  # 这里也要考虑XdG的情况，因为XdG跟任务挂钩所以一旦有了XdG和对应任务都要按照Task-IL的情况来处理
                    x_temp_ = x_[replay_id] if type(x_) == list else x_
                    # 添加XdG的gate
                    if self.mask_dict is not None:
                        self.apply_XdGmask(task=replay_id + 1)
                    y_hat_all = self.classify(x_temp_, not_hidden=replay_not_hidden)

                # if the x_ is not a list, it can't use XdG
                if (not type(x_) == list) and (self.mask_dict is None):
                    y_hat_all = self.classify(x_, not_hidden=replay_not_hidden)

                y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                # calculate loss
                if (y_ is not None) and (y_[replay_id] is not None):
                    predL_re[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='none')
                    predL_re[replay_id] = loss_func.weighted_average(predL_re[replay_id], dim=0)  # take average
                    loss_replay[replay_id] = predL_re[replay_id]
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    # n_class_to_consider = y_hat.size(1)  #针对某些特殊情况需要补正标签的位数
                    distilL_re[replay_id] = loss_func.loss_fn_kd(
                        scores=y_hat, target_scores=scores_[replay_id], T=self.KD_temp
                    )
                    loss_replay[replay_id] = distilL_re[replay_id]

                    # 如果有XdG的情况的话需要在下一个任务的掩码使用之前先反向传播
                    if self.mask_dict is not None:
                        weighted_replay_loss_this_task = (1 - rnt) * loss_replay[replay_id] / n_replays
                        weighted_replay_loss_this_task.backward()

        # Calculate total loss
        loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays
        loss_total = loss_replay if (x is None) else (
            loss_cur if x_ is None else rnt * loss_cur + (1 - rnt) * loss_replay)

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

        ########################################
        # 进行反向传播
        if (self.mask_dict is None) or x_ is None:  # 这里本来是考虑到之前XdG可能会提前反向传播所以需要if判断一下
            loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        optimizer.step()
        functional.reset_net(self)

        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_re': sum(predL_re).item() / n_replays if (x_ is not None and predL_re[0] is not None) else 0,
            'distill_re': sum(distilL_re).item() / n_replays if (x_ is not None and distilL_re[0] is not None) else 0,
            'ewc': ewc_loss.item(),
            'si_loss': surrogate_loss.item(),
            'accuracy': accuracy if accuracy is not None else 0.,
        }


