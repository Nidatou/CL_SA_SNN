import os
import pickle
import torch
from torch import nn

from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from spikingjelly.activation_based import surrogate, neuron

collate_func = None


def checkattr(args, attr):
    return hasattr(args, attr) and getattr(args, attr)


def spiking_collate_fn(batch):
    data, labels = default_collate(batch)
    return torch.transpose(data, 0, 1), labels


def rewind_collate_fn(batch):
    data, labels = default_collate(batch)
    return torch.mean(data, dim=1), labels


def minmax_val(x, dim):
    x_min, _ = torch.min(x, dim=dim, keepdim=True)
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    y = (x - x_min) / (x_max - x_min)
    return y


def get_data_loader(dateset, batch_size, cuda=False, drop_last=False, num_workers=0, shuffle=True):
    """Return <DataLoader>-object for the provided <Dataset>-object"""

    return DataLoader(
        dateset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_func, drop_last=drop_last,
        **({'num_workers': num_workers, 'pin_memory': True} if cuda else{})
    )


def get_neuron_tag(neuron_func):
    if neuron_func == neuron.IFNode:
        neu_tag = 'if'
    elif neuron_func == neuron.LIFNode:
        neu_tag = 'lif'
    else:
        raise NotImplementedError(f"Don't support the neuron function {neuron_func}")

    return neu_tag


def get_surr_tag(surr_func):
    if isinstance(surr_func, surrogate.ATan):
        surr_tag = "ATAN"
    elif isinstance(surr_func, surrogate.Sigmoid):
        surr_tag = "SIGMOID"
    elif isinstance(surr_func, surrogate.SoftSign):
        surr_tag = "SOFTSIGN"
    elif isinstance(surr_func, surrogate.LeakyKReLU):
        surr_tag = "LEAKY"
    elif isinstance(surr_func, surrogate.PiecewiseQuadratic):
        surr_tag = "QUADRATIC"
    else:
        raise NotImplementedError(surr_func)

    return surr_tag


def def_surr_from_tag(neu_tag):
    if neu_tag == 'if':
        neuron_type = neuron.IFNode
    elif neu_tag == 'lif':
        neuron_type = neuron.LIFNode
    else:
        raise NotImplementedError(neu_tag)

    return neuron_type


def def_neuron_from_tag(surr_tag):
    if surr_tag == 'ATAN':
        surr_func = surrogate.ATan()
    elif surr_tag == 'SIGMOID':
        surr_func = surrogate.Sigmoid()
    elif surr_tag == "SOFTSIGN":
        surr_func = surrogate.SoftSign()
    elif surr_tag == "LEAKY":
        surr_func = surrogate.LeakyKReLU()
    elif surr_tag == "QUADRATIC":
        surr_func = surrogate.PiecewiseQuadratic()
    else:
        raise NotImplementedError(surr_tag)

    return surr_func


################################################################
# object-saving and -loading functions #########################
def save_object(object, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def load_object(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)


def count_parameters(model):
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


################################################################
# save/load the model ##########################################
def save_checkpoint(model, model_dir, args=None, verbose=True, name=None):
    name = model.name if name is None else name
    path = os.path.join(model_dir, name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    saved_dict = {}
    saved_dict['state'] = model.state_dict()
    saved_dict['args'] = args
    from models.snn_Adaptive_kWTA import Gate_snn
    from models.SDMLP import SDM
    if isinstance(model, Gate_snn):
        saved_dict['act_total'] = model.topk.last_act_cnts
        print('saved act_total for gate_snn')
    if isinstance(model, SDM) and args.topk is True:
        saved_dict['act_total'] = model.sdm_nets.top_k.neuron_activation_counters
        print('saved act_total for SDM')

    try:
        # torch.save({'state': model.state_dict(),
        #             'args': args}, path)
        torch.save(saved_dict, path)
        if verbose:
            print(f'--> saved model {name} to {model_dir}')
    except OSError:
        print(f'--> saved model {name} failed')


def load_checkpoint(model, model_dir, verbose=True, name=None, add_si_buffers=False, recover_act=True):
    name = model.name if name is None else name
    path = os.path.join(model_dir, name)

    if add_si_buffers:
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                p_current = p.detach().clone()
                omega = p.detach().clone.zero_()
                model.register_buffer(f"{n}_SI_prev_task", p_current)
                model.register_buffer(f'{n}_SI_omega', omega)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'])
    from models.snn_Adaptive_kWTA import Gate_snn
    from models.SDMLP import SDM
    if 'act_total' in checkpoint:
        print('act_total have')
        if isinstance(model, Gate_snn) and recover_act:
            model.topk.last_act_cnts = checkpoint['act_total']
            print('load act_total for Gate_snn')
        elif isinstance(model, SDM) and recover_act:
            model.sdm_nets.top_k.neuron_activation_counters = checkpoint['act_total']
            print('load act_total for SDM')

    if verbose:
        print(f'--> loaded checkpoint of {name} from {model_dir}')


def load_checkpoint_for_attention(model, model_dir, verbose=True, name=None, add_si_buffers=False):
    name = model.name if name is None else name
    path = os.path.join(model_dir, name)

    if add_si_buffers:
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                p_current = p.detach().clone()
                omega = p.detach().clone.zero_()
                model.register_buffer(f"{n}_SI_prev_task", p_current)
                model.register_buffer(f'{n}_SI_omega', omega)

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'])
    # reset the fcLayer's parameters
    if hasattr(model, "layers"):
        for layer_id in range(1, model.layers+1):
            getattr(model, f"fcLayer{layer_id}").apply(weight_reset)

    if verbose:
        print(f'--> loaded checkpoint of {name} from {model_dir}')


################################################################
# parameter initialization function ############################

def weight_reset(m):
    if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)) and m.weight.data.requires_grad:
        m.reset_parameters()


def init_params(model: nn.Module, args):
    model.apply(weight_reset)
    if checkattr(args, 'pre_convE'):
        load_name = model.convE.name if (not hasattr(args, 'convE_ltag') or args.convE_ltag == 'none') \
            else f"{model.convE.name}-{args.convE_ltag}"
        load_checkpoint(model.convE, model_dir=args.model_dir, name=load_name)
        print(f'pretrained conv {load_name} in {args.model_dir} has been loaded')

    if checkattr(args, 'pre_atten'):
        load_name = model.fcPart.name if (not hasattr(args, 'added_ltag') or args.added_ltag == "none") \
            else f"{model.fcPart.name}-{args.added_ltag}"
        load_checkpoint_for_attention(model.fcPart, model_dir=args.model_dir, name=load_name)
        print(f'attention fc {load_name} in {args.model_dir} has been loaded')
    from models.SDMLP import SDM
    from models.snn_Adaptive_kWTA import Gate_snn
    if isinstance(model, SDM):
        model.enforce_weights_regulation()
    # 修改这个的代价可能是无法估量的....
    if isinstance(model, Gate_snn):
        if args.fixed_init:
            model.layer_initialize()  # tmd这个修改的代价才是不可估量的
        model.enforce_weights_regulation()

    return model
