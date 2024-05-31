import argparse
from utils.utils import checkattr

from spikingjelly.activation_based import surrogate, neuron, learning


# Here are several options to use, one can delete one by using [not_include]
# <general> <eval> <task> <model> <train> <replay> <alloc> <perm> <bir>
def handle_inputs(single_task=False, filename="file_name", not_include=list(), description=""):
    kwargs = {'single_task': single_task}
    parser = define_args(filename=filename, description=description)
    parser = add_general_options(parser, **kwargs)
    parser = add_eval_options(parser, **kwargs)
    parser = add_task_options(parser, **kwargs)
    parser = add_model_options(parser, **kwargs)
    parser = add_train_options(parser, **kwargs)
    parser = add_replay_options(parser, **kwargs)
    parser = add_allocation_options(parser, **kwargs) if not_include is None or 'alloc' not in not_include else parser
    parser = add_permutedMNIST_task_options(parser, **kwargs) if not_include is None or 'perm' not in not_include else parser
    parser = add_sdm_options(parser, **kwargs) if not_include is None or 'sdm' not in not_include else parser

    args = parser.parse_args()

    return args


def preprocess_args(args, single_task=False):
    kwargs = {'single_task': single_task}
    translate_for_args(args, **kwargs)
    check_for_errors(args, **kwargs)

    return args


def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser


def boolean_string(s: str) -> bool:
    assert (s in {"False", "True"}), "Not a valid boolean string"
    return s == "True"


def add_general_options(parser, single_task=False, **kwargs):
    parser.add_argument('--save', type=boolean_string, default=True, help="don't save trained models")
    parser.add_argument('--convE_stag', type=str, metavar='STAG', default='none', help="tag for saving convE-layers")
    parser.add_argument('--added_stag', type=str, default='none', help="Added model part tag for added part")
    parser.add_argument('--train', type=boolean_string, default=True, help='evaluate previously saved model')
    if not single_task:
        parser.add_argument('--get_stamp', type=boolean_string, default=False, help='print param-stamp & exit')
    parser.add_argument('--seed', type=int, default=11, help='[first] random seed (for each random-module used)')
    parser.add_argument('--n_seeds', type=int, default=1, help='how often to repeat?')
    # parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
    parser.add_argument('--device', type=str, default='cuda:0', help="The device to deploy model")
    parser.add_argument('--num_workers', type=int, default=0, help='The num_workers of dataloader when cuda used')
    parser.add_argument('--data_dir', type=str, default='./store/datasets', help="default: %(default)s")
    parser.add_argument('--model_dir', type=str, default='./store/opt_models', help="default: %(default)s")
    parser.add_argument('--plot_dir', type=str, default='./store/plots', help="default: %(default)s")
    parser.add_argument('--results_dir', type=str, default='./store/results', help="default: %(default)s")

    return parser


def add_eval_options(parser, single_task=False, **kwargs):
    eval_params = parser.add_argument_group('Evaluation Parameters')
    eval_params.add_argument('--pdf', type=boolean_string, default=False,
                             help="whether to generate pdf with plots for individual experiments")
    if not single_task:
        eval_params.add_argument('--log_per_task', type=boolean_string, default=True)
    eval_params.add_argument('--loss_log', type=int, default=1000, help="# iters after which to log the loss")
    eval_params.add_argument('--acc_log', type=int, default=1000, help="# iters after which to log the accuracy")
    eval_params.add_argument('--acc_n', type=int, default=1024, help="# samples for evaluating accuracy")

    return parser


def add_task_options(parser, single_task=False, **kwargs):
    task_params = parser.add_argument_group("Task parameters")
    if single_task:
        task_choices = ['CIFAR10', 'CIFAR100', 'MNIST', 'MNIST28', 'NMNIST']
        task_default = 'CIFAR10'
    else:
        MNIST_tasks = ['splitMNIST', 'permMNIST', 'splitNMNIST']
        image_tasks = ['CIFAR10', 'CIFAR100', "CIFAR10_Embeddings_0", "CIFAR100_Embeddings"]
        task_choices = MNIST_tasks + image_tasks
        task_default = 'CIFAR10'
    task_params.add_argument('--experiment', type=str, default=task_default, choices=task_choices)
    task_params.add_argument('--no_norm', type=boolean_string, default=False)
    task_params.add_argument('--augment', type=boolean_string, default=True)
    if not single_task:
        # 'task':   each task has own output-units, always only those units are considered
        # 'domain': each task is mapped to the same output-units
        # 'class':  each task has own output-units, all units of tasks seen so far are considered
        task_params.add_argument('--scenario', type=str, default='task', choices=['task', 'domain', 'class'])
        task_params.add_argument('--tasks', type=int, help='number of tasks')
    return parser


def add_permutedMNIST_task_options(parser, single_task=False, **kwargs):
    perm_params = parser.add_argument_group("Perm_task Parameters")
    perm_params.add_argument('--tasks', type=int, help='number of permutations')

    return parser


def add_model_options(parser, single_task=False, **kwargs):
    model_params = parser.add_argument_group('Parameters for SNN model')
    model_params.add_argument('--conv_type', type=str, default="standard", choices=["standard", "resNet", 'vggsnn'])
    model_params.add_argument('--n_blocks', type=int, default=2, help="# blocks per conv-layer (only for 'resNet')")
    model_params.add_argument('--depth', type=int, help="# of convolutional layers (0 = only fc-layers)")
    model_params.add_argument('--reducing_layers', type=int, dest='rl',
                              help="# of layers with stride (=image-size halved)")
    model_params.add_argument('--channels', type=int, default=16,
                              help="# of channels 1st conv-layer (doubled every 'rl')")
    model_params.add_argument('--conv_bn', type=boolean_string, default=True,
                              help="use batch-norm in the conv-layers (yes|no)")
    model_params.add_argument('--conv_neu', type=str, default="IF", choices=["LIF", "IF"])
    model_params.add_argument('--global_pooling', action='store_true', dest='gp',
                              help="ave global pool after conv-layers")

    model_params.add_argument('--ann_vae', type=boolean_string, default=False, help="whether to use ann variant encoder")
    model_params.add_argument('--fc_depth', type=int, default=3, help="# of fully-connected layers")
    model_params.add_argument('--fc_units', type=int, default=2000 if single_task else None, help='# of hidden units')
    model_params.add_argument('--fc_drop', type=float, default=0., help='dropout probability of fc layers')
    model_params.add_argument('--fc_neu', type=str, default='IF', help='the neurons type of SNN')
    model_params.add_argument('--h_dim', type=int, help='# of final features units')

    # attention gate part
    model_params.add_argument('--atten_type', type=str, default='normal', choices=['normal', 'recurrent'],
                              help="the attention type of attention layer")

    # SNN biology part
    model_params.add_argument('--T', type=int, default=16, help='the time steps of the SNN')
    model_params.add_argument('--surrogate', type=str, default='ATAN', help='the surrogate function of the spiking')
    model_params.add_argument('--tau', type=float, default=2., help='the constant parameter for LIF neuron')

    return parser


def add_train_options(parser, single_task=False, **kwargs):
    train_params = parser.add_argument_group('Training Parameters')
    if single_task:
        iter_epochs = train_params.add_mutually_exclusive_group(required=False)
        iter_epochs.add_argument('--epochs', type=int, default=10, )
        iter_epochs.add_argument('--iters', type=int, help='max # of iterations (replace "--epochs")')
    else:
        train_params.add_argument('--iters', type=int, help='# batches to optimize main model')
        train_params.add_argument('--iters_first', type=int, help='# batches to train the first task')

    # if single task?
    train_params.add_argument('--optimiser', type=str, default='sgd', choices=['adam', 'sgd', 'sgdm'], help="the type of optimizer")
    train_params.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    train_params.add_argument('--batch', type=int, default=256 if single_task else None, help='batch size of the dataloader')

    train_params.add_argument('--convE_ltag', type=str, default='none', help="tag for loading the convE-layers")
    train_params.add_argument('--pre_convE', type=boolean_string, default=False, help='use pretrained feature-extractor')
    train_params.add_argument('--freeze_convE', type=boolean_string, default=False, help='freeze parameter of feature extractor')

    train_params.add_argument('--added_ltag', type=str, default='none', help="tag for loading the added part")
    train_params.add_argument('--pre_atten', type=boolean_string, default=False, help='use pretrained attention')
    train_params.add_argument('--freeze_atten', type=boolean_string, default=False, help='freeze parameter of attention part')
    train_params.add_argument('--fixed_init', type=boolean_string, default=False, help='whether to fixed initialize for SA-SNN')

    train_params.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'tet'])
    train_params.add_argument('--loss_lambda', type=float, default=1e-3, help='the parameter lambda for <tet> loss function')
    train_params.add_argument('--loss_means', type=float, default=1., help='the parameters for <tet> regularization part')
    train_params.add_argument('--record_process', type=boolean_string, default=True)

    return parser


def add_replay_options(parser, **kwargs):
    replay_params = parser.add_argument_group('Replay Parameters')
    replay_choices = ['offline', 'generative', 'none', 'current']
    replay_params.add_argument('--replay', type=str, default='none', choices=replay_choices)
    replay_params.add_argument('--distill', type=boolean_string, default=False, help="whether to use distillation")
    replay_params.add_argument('--temp', type=float, default=2., help='the temperature for distillation')

    return parser


def add_allocation_options(parser, **kwargs):
    cl = parser.add_argument_group('Memory Allocation Parameters')

    cl.add_argument('--model_plat', type=str, default='standard', choices=['standard', 'flymodel', 'sleep', 'test'])

    cl.add_argument('--grad_clip', type=float, default=1., help='the clipped value of the gradient')
    cl.add_argument('--fisher_n', type=int, default=1000, help='--> EWC: number of samplers to generate Fisher Information')
    cl.add_argument('--epsilon', type=float, default=0.1, help='--> SI: dampening parameter')

    cl.add_argument('--ewc', type=boolean_string, default=False, help='use EWC')
    cl.add_argument('--online', type=boolean_string, default=False, help='perform online EWC')
    cl.add_argument('--si', type=boolean_string, default=False, help='use SI')
    cl.add_argument('--xdg', type=boolean_string, default=False, help='use XdG')

    cl.add_argument('--o-lambda', type=float, help='--> online EWC: regularisation strength')
    cl.add_argument('--ewc_lambda', type=float, help='--> EWC: regularization strength')
    cl.add_argument('--ewc_beta', type=float, default=1., help='params in specific EWC method')
    cl.add_argument('--simpled_EWC', type=boolean_string, default=False, help='whether to use a simplified EWC version')
    cl.add_argument('--cl_batch_size', type=int, default=512, help='the fisher training batch size')
    cl.add_argument('--cl_batches', type=int, default=5, help='the fisher training num of batches')

    cl.add_argument('--gamma', type=float, help='EWC: forgetting coefficient (for online EWC)')
    cl.add_argument('--si_c', type=float, help='--> SI: regularization strength')
    cl.add_argument('--xdg_prop', type=float, help='--> XdG: prop neuron per layer to gate')

    # parameter for MAS method
    cl.add_argument('--mas', type=boolean_string, default=False)
    cl.add_argument('--online_mas', type=boolean_string, default=True)
    cl.add_argument('--mas_lambda', type=float, default=0.)

    # parameter for flymodels
    cl.add_argument('--n_kc', type=int, default=1000)
    cl.add_argument('--kc_response', type=int, default=32)
    cl.add_argument('--k_num', type=int, default=32)
    cl.add_argument('--min_max', type=boolean_string, default=True)

    return parser


def add_sdm_options(parser, **kwargs):
    topk_params = parser.add_argument_group('SDM Parameters')
    # -> topk parameters
    topk_params.add_argument('--topk', type=boolean_string, default=True)
    topk_params.add_argument('--use_ann', type=boolean_string, default=False)
    topk_params.add_argument('--fnl', type=str, choices=['mem', 'normal'], default='normal')
    #######################################################################
    # the parameters concerning about WTA-k method itself
    topk_params.add_argument('--k_max', type=int, default=None)
    topk_params.add_argument('--k_min', type=int, default=10)
    topk_params.add_argument('--k_trans_ep', type=int, default=50)
    # topk method
    topk_params.add_argument('--use_mask', type=boolean_string, default=True)
    topk_params.add_argument('--inhibit_intense', type=float, default=1.)
    topk_params.add_argument('--single_WTA', type=boolean_string, default=False)
    topk_params.add_argument('--detach_nonspike', type=boolean_string, default=False)
    topk_params.add_argument('--gaba_switch_num', type=int, default=5000000)
    topk_params.add_argument('--k_param', type=int, default=100)
    topk_params.add_argument('--k_approach', type=str, default="FLAT",
                             choices=["GABA_SWITCH", "LINEAR_DECAY", "FLAT"])

    # about adaptive threshold
    topk_params.add_argument('--adaptive_thresh', type=boolean_string, default=True)
    topk_params.add_argument('--adap_param', type=int, default=int(1e+5))
    topk_params.add_argument('--adap_approach', type=str, default="linear", choices=['linear', 'exponential'])
    topk_params.add_argument('--thresh_max', type=float, default=2.0)
    topk_params.add_argument('--thresh_min', type=float, default=1.0)

    #######################################################################
    # the parameters of WTA-k augment method itself
    topk_params.add_argument('--last_inhibit', type=boolean_string, default=False)
    topk_params.add_argument('--inhibit_p', type=float, default=1e-4)
    topk_params.add_argument('--curr_excite', type=boolean_string, default=False)
    topk_params.add_argument('--excite_p', type=float, default=1e-5)
    topk_params.add_argument('--tune_approach', type=str, default="sigmoid", choices=['sigmoid', 'linear', 'tanh'])
    topk_params.add_argument('--tune_param', type=float, default=1.0)

    topk_params.add_argument('--adap_curr', type=boolean_string, default=True)

    # topk_params.add_argument('--nneu', type=int, default=1000)
    topk_params.add_argument('--neu', type=str, default='if', choices=['if', 'lif'])
    # -> weight normalization
    topk_params.add_argument('--dale', type=boolean_string, default=True)
    topk_params.add_argument('--norm_ad', type=boolean_string, default=True)
    topk_params.add_argument('--norm_val', type=boolean_string, default=False)
    topk_params.add_argument('--norm_input', type=boolean_string, default=True)

    #######################################################################
    # the parameters appended for trace mask
    topk_params.add_argument('--step_mask', type=boolean_string, default=False)

    topk_params.add_argument('--hard_mask', type=boolean_string, default=False)
    topk_params.add_argument('--lateral_inh', type=boolean_string, default=False)
    topk_params.add_argument('--self_inh', type=boolean_string, default=False)
    # -> parameters about trace
    topk_params.add_argument('--tr_decay', type=float, default=10.)
    topk_params.add_argument('--tr_refresh', type=boolean_string, default=True)
    topk_params.add_argument('--loosen_tr', type=boolean_string, default=False)
    topk_params.add_argument('--one_size_tr', type=boolean_string, default=False)

    #######################################################################
    # added parameters
    topk_params.add_argument('--spk_ipt', type=boolean_string, default=False, help="whether set the input mode")
    topk_params.add_argument('--filter_prop', type=float, default=0.1)
    topk_params.add_argument('--norm_const', type=float, default=1.)

    #######################################################################
    # test for spk
    topk_params.add_argument('--need_sleep', type=boolean_string, default=False)
    topk_params.add_argument('--topk_sleep', type=boolean_string, default=False)
    topk_params.add_argument('--inc_lr', type=float, default=1e-4)
    topk_params.add_argument('--dec_lr', type=float, default=1e-4)
    topk_params.add_argument('--sleep_times', type=int, default=64)
    topk_params.add_argument('--sleep_batch', type=int, default=128)
    topk_params.add_argument('--weight_lim', type=boolean_string, default=True)

    return parser


# modify the options
def translate_for_args(args, single_task=False, **kwargs):
    if hasattr(args, 'fc_neu'):
        if args.fc_neu == 'LIF':
            args.fc_neu = neuron.LIFNode
        elif args.fc_neu == 'IF':
            args.fc_neu = neuron.IFNode
        else:
            raise NotImplementedError(args.fc_neu)
    if hasattr(args, 'conv_neu'):
        if args.conv_neu == 'LIF':
            args.conv_neu = neuron.LIFNode
        elif args.conv_neu == 'IF':
            args.conv_neu = neuron.IFNode
        else:
            raise NotImplementedError(args.conv_neu)

    # Get the surrogate function for neurons
    if hasattr(args, 'surrogate'):
        args.surrogate = args.surrogate.upper()
        if args.surrogate == 'ATAN':
            args.surrogate = surrogate.ATan()
        elif args.surrogate == 'SIGMOID':
            args.surrogate = surrogate.Sigmoid()
        elif args.surrogate == "SOFTSIGN":
            args.surrogate = surrogate.SoftSign()
        elif args.surrogate == "LEAKY":
            args.surrogate = surrogate.LeakyKReLU()
        elif args.surrogate == "QUADRATIC":
            args.surrogate = surrogate.PiecewiseQuadratic()
        else:
            raise NotImplementedError(args.surrogate)

    # -> take the model without normalization into account
    if "MNIST" in args.experiment or 'Embeddings' in args.experiment:
        args.no_norm = True
        args.augment = False
        args.freeze_convE = False

    if hasattr(args, "depth"):
        args.depth = (5 if args.experiment in ('CIFAR10', 'CIFAR100') else 0) if args.depth is None else args.depth
        if "embeddings" in args.experiment.lower():
            args.pre_convE = False
            args.depth = 0
    if hasattr(args, "recon_loss"):
        args.recon_loss = (
            "MSE" if args.experiment in ('CIFAR10', 'CIFAR100') else "BCE"
        ) if args.recon_loss is None else args.recon_loss
    if hasattr(args, "dg_type"):
        args.dg_type = ("task" if args.experiment == 'permMNIST' else "class") if args.dg_type is None else args.dg_type

    if not single_task:
        args.tasks = (
            5 if args.experiment=='splitMNIST' or 'CIFAR10' in args.experiment else (10 if "CIFAR100" in args.experiment else 5)
        ) if args.tasks is None else args.tasks
        args.iters = (5000 if args.experiment=='CIFAR100' else 2000) if args.iters is None else args.iters
        args.lr = 1e-4 if args.lr is None else args.lr
        args.batch = (128 if args.experiment=='splitMNIST' else 256) if args.batch is None else args.batch
        # args.fc_units = (400 if args.experiment=='splitMNIST' else 2000) if args.fc_units is None else args.fc_units
        args.fc_units = 1000 if args.fc_units is None else args.fc_units
    else:
        args.fc_unints = (400 if "MNIST" in args.experiment else 2000) if args.fc_units is None else args.fc_units
        args.batch = (128 if "MNIST" in args.experiment else 2000) if args.fc_units is None else args.batch
        args.lr = 1e-4 if args.lr is None else args.lr

    args.h_dim = args.fc_units if args.h_dim is None else args.h_dim

    if not single_task:
        if args.experiment == 'splitMNIST':
            args.xdg_prop = 0.9 if args.scenario=='task' and args.xdg_prop is None else args.xdg_prop
            args.si_c = (100. if args.scenario=='task' else 0.1) if args.si_c is None else args.si_c
            args.ewc_lambda = (
                1000000000. if args.scenario=='task' else 100000.
            ) if args.ewc_lambda is None else args.ewc_lambda
            args.gamma = 1. if args.gamma is None else args.gamma
            if hasattr(args, 'dg_prop'):
                args.dg_prop = 0.8 if args.dg_prop is None else args.dg_prop
        elif args.experiment == 'CIFAR100' or args.experiment == "CIFAR10":
            args.xdg_prop = 0.7 if args.scenario=='task' and args.xdg_prop is None else args.xdg_prop
            args.si_c = (100. if args.scenario=='task' else 1.) if args.si_c is None else args.si_c
            args.ewc_lambda = (1000. if args.scenario=='task' else 1.) if args.ewc_lambda is None else args.ewc_lambda
            args.gamma = 1 if args.gamma is None else args.gamma
            # args.dg_prop = (0. if args.scenario=='task' else 0.7) if args.dg_prop is None else args.dp_prop
            # args.dg_si_prop = 0.6 if args.dg_si_prop is None else args.dg_si_prop
            # args.dg_c = 100000000. if args.dg_c is None else args.dg_c
        elif args.experiment == 'permMNIST':
            args.si_c = 10. if args.si_c is None else args.si_c
            args.ewc_lambda = 1. if args.ewc_lambda is None else args.ewc_lambda
            if hasattr(args, 'o_lambda'):
                args.o_lambda = 1. if args.o_lambda is None else args.o_lambda
            args.gamma = 1. if args.gamma is None else args.gamma
            args.dg_prop = 0.8 if args.dg_prop is None else args.dg_prop
            args.dg_si_prop = 0.8 if args.dg_si_prop is None else args.dg_si_prop
            args.dg_c = 1. if args.dg_c is None else args.dg_c
        # -for other unselected options, set default values (not specific to chosen scenario / experiment)

    if hasattr(args, "lr_gen"):
        args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    if hasattr(args, "rl"):
        args.rl = args.depth - 1 if args.rl is None else args.rl
    if not single_task:
        args.xdg_prop = 0. if args.scenario == "task" and args.xdg_prop is None else args.xdg_prop
    # -if [log_per_task] (which is default for comparison-scripts), reset all logs
    if checkattr(args, 'log_per_task'):
        args.acc_log = args.iters
        args.loss_log = args.iters

    return args


def check_for_errors(args, single_task=False, ** kwargs):
    if not single_task:
        # -if scenario is "class" and XdG is selected, give error
        if args.scenario == "class" and checkattr(args, 'xdg') and args.xdg_prop > 0:
            raise ValueError("Having scenario=[class] with 'XdG' does not make sense")
        # -if scenario is "domain" and XdG is selected, give warning
        if args.scenario == "domain" and checkattr(args, 'xdg') and args.xdg_prop > 0:
            print("Although scenario=[domain], 'XdG' makes that task identity is nevertheless always required")
        # -if XdG is selected together with replay of any kind, give error
        if checkattr(args, 'xdg') and args.xdg_prop > 0 and (not args.replay == "none"):
            raise NotImplementedError("XdG is not supported with '{}' replay.".format(args.replay))
            # --> problem is that applying different task-masks interferes with gradient calculation
            #    (should be possible to overcome by calculating each gradient before applying next mask)
        # -if 'only_last' is selected with replay, EWC or SI, give error
        if checkattr(args, 'only_last') and (not args.replay == "none"):
            raise NotImplementedError("Option 'only_last' is not supported with '{}' replay.".format(args.replay))
        if checkattr(args, 'only_last') and (checkattr(args, 'ewc') and args.ewc_lambda > 0):
            raise NotImplementedError("Option 'only_last' is not supported with EWC.")
        if checkattr(args, 'only_last') and (checkattr(args, 'si') and args.si_c > 0):
            raise NotImplementedError("Option 'only_last' is not supported with SI.")
        # -error in type of reconstruction loss
    if not args.no_norm and hasattr(args, "recon_los") and args.recon_loss == "BCE":
        raise ValueError("'BCE' is not a valid reconstruction loss with normalized images")

