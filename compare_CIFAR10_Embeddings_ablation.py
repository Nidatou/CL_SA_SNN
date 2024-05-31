#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
from utils.param_stamps import get_param_stamp_from_args
from utils import options, utils
from visual import plotting
import main_exp


def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile(f"{args.results_dir}/dict-{param_stamp}.pkl"):
        print(f"{param_stamp}: already run")
    else:
        print(f"{param_stamp}: ...running...")
        main_exp.run(args, verbose=False)
    # -get average accuracies
    fileName = f'{args.results_dir}/acc-{param_stamp}.txt'
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -results-dict
    dict = utils.load_object(f"{args.results_dir}/dict-{param_stamp}")
    # -return tuple with the results
    return dict, ave


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict


if __name__ == '__main__':

    args = options.handle_inputs(
        single_task=False, not_include=["perm"], filename="_compare_CIFAR10",
        description="Compare performance of continual learning strategies on different scenarios of split CIFAR-10."
    )

    ################################################################################
    # Add default arguments (will be different for different runs)
    args.experiment = "CIFAR10_Embeddings_0"
    # args.experiment = "splitMNIST"
    args.scenario = 'class'
    args.replay = "none"
    args.distill = False
    args.feedback = False
    args.hidden = False
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = False
    args.topk = False
    args.freeze_convE = False
    args.iters = 20000
    # args.loss_type = 'tet'

    # Use pre-trained convolutional layers for all compared methods
    # args.pre_convE = True
    # args.freeze_convE = True
    args.pre_convE = False
    args.freeze_convE = False
    args.augment = True
    args.no_norm = False
    args.surrogate = "ATAN"
    args.fc_depth = 2
    args.fc_units = 1000
    args.h_dim = 1000
    args.lr = 5e-2
    args.optimiser = 'sgd'

    args.log_per_task = False
    args.acc_n = None
    args.loss_log = 20000
    args.acc_log = 2000
    args.seed = 12
    args.n_seeds = 3
    args = options.preprocess_args(args, single_task=False)

    # If needed, create plotting directory
    if not os.path.isdir(args.plot_dir):
        os.mkdir(args.plot_dir)

    print(f'acc_log:{args.acc_log}')

    #########################################################
    # -> ALL METHODS ##############################
    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    ###########################################################
    # All kinds of ablations methods ########
    args.topk = True
    args.use_mask = True
    args.hard_mask = False
    args.use_ann = False

    # SNN_step + EWC #####################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True
    args.adaptive_thresh = True
    args.adap_param = 2000000
    args.k_min = 10
    args.norm_const = 0.25
    args.ewc = True
    args.simpled_EWC = True
    args.ewc_lambda = 100
    args.ewc_beta = 0.08
    WTK_S_EWC = {}
    WTK_S_EWC = collect_all(WTK_S_EWC, seed_list, args, name="WTK_S_EWC")

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.
    args.ewc = False

    # SNN_step + EWC + w/o thresh  #####################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True
    args.adap_param = 2000000
    args.k_min = 10
    args.norm_const = 0.25
    args.ewc = True
    args.simpled_EWC = True
    args.ewc_lambda = 100
    args.ewc_beta = 0.08
    args.adaptive_thresh = False
    args.thresh_min = 1.
    WTK_S_EWC_solid_1 = {}
    WTK_S_EWC_solid_1 = collect_all(WTK_S_EWC_solid_1, seed_list, args, name="WTK_S_EWC_solid_1")
    dict = WTK_S_EWC_solid_1[seed_list[0]][0]
    for key in dict.keys():
        print(f"{key}: {type(dict[key][0])}")

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.
    args.adaptive_thresh = True
    args.ewc = False

    # SNN_mask #################################################
    args.use_mask = True
    args.topk = True
    args.step_mask = False
    args.use_ann = False
    args.adap_param = 2000000
    WTK_M = {}
    WTK_M = collect_all(WTK_M, seed_list, args, name="WTK_M")

    args.use_mask = True

    # SNN_step ###########################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True
    args.norm_const = 0.25
    args.adap_param = 2000000
    args.k_min = 10
    WTK_S = {}
    WTK_S = collect_all(WTK_S, seed_list, args, name="WTK_S")

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.

    # SNN_step - w/o thresh  #####################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True
    args.adap_param = 2000000
    args.k_min = 10
    args.norm_const = 0.25
    args.adaptive_thresh = False
    args.thresh_min = 1.
    WTK_S_solid_1 = {}
    WTK_S_solid_1 = collect_all(WTK_S_solid_1, seed_list, args, name="WTK_S_solid_1")

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.
    args.adaptive_thresh = True

    # SNN_step - w/o dale  #####################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True
    args.adap_param = 2000000
    args.k_min = 10
    args.norm_const = 0.25
    args.adaptive_thresh = True
    args.dale = False
    args.norm_input = True
    args.norm_ad = True
    args.thresh_min = 1.
    WTK_S_no_dale = {}
    WTK_S_no_dale = collect_all(WTK_S_no_dale, seed_list, args, name="WTK_S_no_dale")

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.
    args.adaptive_thresh = True
    args.dale = True
    args.norm_input = True
    args.norm_ad = True

    # SNN_step - w/o Top-K  #####################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = False
    args.adap_param = 2000000
    args.k_min = 1000
    args.norm_const = 0.25
    args.adaptive_thresh = True
    args.dale = True
    args.norm_input = True
    args.norm_ad = True
    args.thresh_min = 1.
    SNN_no_WTK = {}
    SNN_no_WTK = collect_all(SNN_no_WTK, seed_list, args, name="SNN_no_WTK")

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.
    args.adaptive_thresh = True
    args.dale = True
    args.norm_input = True
    args.norm_ad = True
    args.step_mask = True

    # SNN_step - w/o norm  #####################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True
    args.adap_param = 2000000
    args.k_min = 10
    args.norm_const = 0.25
    args.adaptive_thresh = True
    args.dale = True
    args.norm_input = False
    args.norm_ad = False
    args.thresh_min = 1.
    WTK_S_no_norm = {}
    WTK_S_no_norm = collect_all(WTK_S_no_norm, seed_list, args, name="WTK_S_no_norm")

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.
    args.adaptive_thresh = True
    args.dale = True
    args.norm_input = True
    args.norm_ad = True

    WITH_EWC = [WTK_S_EWC, WTK_S_EWC_solid_1, WTK_S_no_norm, WTK_S_no_norm, WTK_S_no_norm, WTK_S_no_norm]
    SNN_STEP = [WTK_S, WTK_M, WTK_S_solid_1, WTK_S_no_dale, WTK_S_no_norm, SNN_no_WTK]

    ###########################################################
    # PLOTTING ########

    # print header to screen
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    print("\n\n" + "#" * 60 + "\nSUMMARY RESULTS: {}\n".format(title) + "#" * 60)

    plot_name = f"Ablation_{args.experiment}{args.tasks}-{args.scenario}"
    names_1 = ["S+E", '-adp th', '', '', '', '']
    colors_1 = ['white', '#E69F00', 'white', 'white', 'white', 'white']
    names_2 = ['Step', 'Mask', '-adp th', '-dale', '-norm', '-topk']
    colors_2 = ['white', '#0072B2', '#E69F00', '#009E73', '#F0E442', '#CC79A7']

    # ['#0072B2', '#E69F00', '#009E73', '#F0E442', '#CC79A7', '#56B4E9', '#D55E00', '#999999']

    total_classes = args.tasks * int(np.floor(100/args.tasks))

    ylabel = "Test accuracy"
    title = "AVERAGE ACCURACY (in %)"
    chance_level = (100. / total_classes) if args.scenario == 'class' else (100./int(np.floor(100/args.tasks)))
    print("\n\n{}\n".format(title) + "-" * 60)
    figure = plotting.plot_ablation(
        dict_list1=WITH_EWC, dict_list2=SNN_STEP, names_1=names_1, names_2=names_2,
        colors_1=colors_1, colors_2=colors_2, seed_list=seed_list, index=1,
        change_level=chance_level, ylabel=ylabel, title=title, ylim=[0, 0.93],
    )
    plt.savefig(os.path.join(args.plot_dir, plot_name), dpi=300)
