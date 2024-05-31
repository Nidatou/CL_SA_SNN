#!/usr/bin/env python3
import os
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
        # main_exp.run(args, verbose=False)
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


def add_zeroInacc(method_dict, seed_list, task_num):
    for seed in seed_list:
        src_num = len(method_dict[seed][0]["all_tasks"][0])
        for task_id in range(task_num - 1):
            curr_num = len(method_dict[seed][0]["all_tasks"][task_id + 1])
            added_zero = [0] * (src_num - curr_num)
            method_dict[seed][0]["all_tasks"][task_id + 1] = added_zero + method_dict[seed][0]["all_tasks"][task_id + 1]


if __name__ == '__main__':

    args = options.handle_inputs(
        single_task=False, not_include=["perm"], filename="_compare_CIFAR10",
        description="Compare performance of continual learning strategies on different scenarios of split CIFAR-10."
    )

    ################################################################################
    # Add default arguments (will be different for different runs)
    args.experiment = "CIFAR100_Embeddings"
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
    args.iters = 4000
    args.tasks = 25
    args.T = 16
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
    args.fc_units = 2000
    args.h_dim = 2000
    args.lr = 5e-2
    args.optimiser = 'sgd'

    args.log_per_task = False
    args.acc_n = None
    args.loss_log = 4000
    args.acc_log = 4000
    args.seed = 12
    args.n_seeds = 3
    args.record_process = False
    args = options.preprocess_args(args, single_task=False)

    # If needed, create plotting directory
    if not os.path.isdir(args.plot_dir):
        os.mkdir(args.plot_dir)

    print(f'acc_log:{args.acc_log}')

    #########################################################
    # -> ALL METHODS ##############################
    seed_list = list(range(args.seed, args.seed+args.n_seeds))

    ###########################################################
    # Baseline ########
    print(args.scenario)
    # Joint
    args.replay = "offline"
    args.topk = False
    args.fc_units = 2000
    args.h_dim = 2000

    OFF_2K = {}
    OFF_2K = collect_all(OFF_2K, seed_list, args, name="Joint_2K")

    args.fc_units = 10000
    args.h_dim = 10000

    OFF_10K = {}
    OFF_10K = collect_all(OFF_10K, seed_list, args, name="Joint_10K")

    # None
    args.replay = "none"
    args.fc_units = 10000
    args.h_dim = 10000
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")

    args.replay = "none"

    ###########################################################
    # Other methods ########
    args.topk = True
    args.use_mask = True
    args.hard_mask = False
    args.use_ann = False
    args.norm_const = 1.
    ###########################################################
    # FlyModel ################################################
    ###########################################################
    args.model_plat = 'flymodel'
    args.iters = 8
    args.acc_log = 8
    args.n_kc = 2000
    args.k_num = 20
    args.lr = 0.2
    args.min_max = True
    args.kc_response = 64
    args.record_process = False

    FLY_2K = {}
    FLY_2K = collect_all(FLY_2K, seed_list, args, name="FLY_1K")

    args.n_kc = 10000
    args.lr = 0.2
    args.k_num = 100
    args.kc_response = 64
    FLY_10K = {}
    FLY_10K = collect_all(FLY_10K, seed_list, args, name="FLY_10K")

    args.model_plat = 'standard'
    args.iters = 4000
    args.acc_log = 4000
    args.lr = 5e-2
    args.record_process = False

    # SNN_mask #################################################
    args.use_mask = True
    args.topk = True
    args.step_mask = False
    args.use_ann = False

    args.fc_units = 2000
    args.h_dim = 2000
    args.k_min = 20
    args.adap_param = 2000000
    args.iters = 4000
    args.iters_first = 40000
    WTK_M_2K = {}
    WTK_M_2K = collect_all(WTK_M_2K, seed_list, args, name="WTK_M_2K")

    args.fc_units = 10000
    args.h_dim = 10000
    args.adap_param = 1000000
    args.k_min = 20
    args.iters = 4000
    args.iters_first = 40000
    WTK_M_10K = {}
    WTK_M_10K = collect_all(WTK_M_10K, seed_list, args, name="WTK_M_10K")

    args.iters_first = None
    args.use_mask = True

    # SNN_mask + EWC ##########################################
    args.use_mask = True
    args.topk = True
    args.step_mask = False
    args.use_ann = False
    args.ewc = True
    args.simpled_EWC = True
    args.ewc_lambda = 100
    args.ewc_beta = 0.08

    args.fc_units = 2000
    args.h_dim = 2000
    args.k_min = 20
    args.adap_param = 2000000
    args.iters = 4000
    args.iters_first = 40000

    WTK_M_EWC_2K = {}
    WTK_M_EWC_2K = collect_all(WTK_M_EWC_2K, seed_list, args, name="WTK_M_EWC_2K")

    args.ewc = True
    args.simpled_EWC = True
    args.ewc_lambda = 100
    args.ewc_beta = 0.08

    args.fc_units = 10000
    args.h_dim = 10000
    args.k_min = 20
    args.adap_param = 1000000
    args.iters = 4000
    args.iters_first = 40000
    WTK_M_EWC_10K = {}
    WTK_M_EWC_10K = collect_all(WTK_M_EWC_10K, seed_list, args, name="WTK_M_EWC_10K")

    args.iters_first = None
    args.use_mask = True
    args.ewc = False

    # SNN_step ###########################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True

    args.fc_units = 2000
    args.h_dim = 2000
    args.k_min = 20
    args.adap_param = 1000000
    args.iters = 4000
    args.iters_first = 12000
    args.norm_const = 0.25

    WTK_S_2K = {}
    WTK_S_2K = collect_all(WTK_S_2K, seed_list, args, name="WTK_S_2K")

    args.fc_units = 10000
    args.h_dim = 10000
    args.k_min = 40
    args.adap_param = 1000000
    args.iters = 8000
    args.acc_log = 8000
    args.iters_first = 40000
    args.norm_const = 0.25

    WTK_S_10K = {}
    WTK_S_10K = collect_all(WTK_S_10K, seed_list, args, name="WTK_S_10K")

    args.iters = 4000
    args.acc_log = 4000
    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.
    args.iters_first = None

    # SNN_step + EWC #####################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True

    args.ewc = True
    args.simpled_EWC = True
    args.ewc_lambda = 100
    args.ewc_beta = 0.08

    args.fc_units = 2000
    args.h_dim = 2000
    args.k_min = 20
    args.adap_param = 1000000
    args.iters = 4000
    args.iters_first = 12000
    args.norm_const = 0.25
    args.lr = 0.08
    WTK_S_EWC_2K = {}
    WTK_S_EWC_2K = collect_all(WTK_S_EWC_2K, seed_list, args, name="WTK_S_EWC_2K")

    args.fc_units = 10000
    args.h_dim = 10000
    args.k_min = 40
    args.adap_param = 1000000
    args.iters = 8000
    args.acc_log = 8000
    args.iters_first = 40000
    args.norm_const = 0.25
    args.lr = 0.05
    WTK_S_EWC_10K = {}
    WTK_S_EWC_10K = collect_all(WTK_S_EWC_10K, seed_list, args, name="WTK_S_EWC_10K")

    args.iters = 4000
    args.acc_log = 4000
    args.ewc = False
    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.
    args.iters_first = None

    # EWC
    args.ewc = True
    args.simple_EWC = True
    args.ewc_lambda = 200
    args.ewc_beta = 0.08
    args.topk = False
    args.use_ann = False

    args.fc_units = 2000
    args.h_dim = 2000
    args.iters = 4000
    EWC_2K = {}
    EWC_2K = collect_all(EWC_2K, seed_list, args, name="EWC_2K")

    args.fc_units = 10000
    args.h_dim = 10000
    args.iters = 4000
    EWC_10K = {}
    EWC_10K = collect_all(EWC_10K, seed_list, args, name="EWC_10K")

    args.ewc = False
    args.topk = True

    # MAS
    args.mas = True
    args.mas_lambda = 40.
    args.topk = False
    args.use_ann = False

    args.fc_units = 2000
    args.h_dim = 2000
    args.iters = 4000
    MAS_2K = {}
    MAS_2K = collect_all(MAS_2K, seed_list, args, name="MAS_2K")

    args.fc_units = 10000
    args.h_dim = 10000
    args.iters = 4000
    MAS_10K = {}
    MAS_10K = collect_all(MAS_10K, seed_list, args, name="MAS_10K")

    args.mas = False
    args.topk = True

    # # SI
    # args.si = True
    # SI = {}
    # SI = collect_all(SI, seed_list, args, name="SI")
    # args.si = False

    ###########################################################
    # COLLECT RESULTS ########
    acc_2K = {}
    task_acc_2K = {}
    ave_acc_2K = {}

    acc_10K = {}
    task_acc_10K = {}
    ave_acc_10K = {}

    # Create lists for all extracted <dicts> and <lists> with fixed order
    task_num = args.tasks
    line_size = len(OFF_2K[seed_list[0]][0]['average'])
    for seed in seed_list:
        i = 0
        acc_2K[seed] = [
            OFF_2K[seed][i]["average"][-line_size:], NONE[seed][i]["average"][-line_size:],
            EWC_2K[seed][i]['average'][-line_size:],
            MAS_2K[seed][i]['average'][-line_size:],
            FLY_2K[seed][i]["average"][-line_size:],
            # SDM[seed][i]["average"],
            WTK_M_2K[seed][i]['average'][-line_size:],
            WTK_S_2K[seed][i]['average'][-line_size:],
            WTK_M_EWC_2K[seed][i]['average'][-line_size:],
            # SDM_EWC[seed][i]["average"],
            WTK_S_EWC_2K[seed][i]['average'][-line_size:],
        ]
        acc_10K[seed] = [
            OFF_10K[seed][i]["average"][-line_size:], NONE[seed][i]["average"][-line_size:],
            EWC_10K[seed][i]['average'][-line_size:],
            MAS_10K[seed][i]['average'][-line_size:],
            FLY_10K[seed][i]["average"][-line_size:],
            # SDM[seed][i]["average"],
            WTK_M_10K[seed][i]['average'][-line_size:],
            WTK_S_10K[seed][i]['average'][-line_size:],
            WTK_M_EWC_10K[seed][i]['average'][-line_size:],
            # SDM_EWC[seed][i]["average"],
            WTK_S_EWC_10K[seed][i]['average'][-line_size:],
        ]
        i = 1
        ave_acc_2K[seed] = [
            OFF_2K[seed][i], NONE[seed][i],
            EWC_2K[seed][i],
            MAS_2K[seed][i],
            FLY_2K[seed][i],
            # SDM[seed][i],
            WTK_M_2K[seed][i],
            WTK_S_2K[seed][i],
            WTK_M_EWC_2K[seed][i],
            # SDM_EWC[seed][i],
            WTK_S_EWC_2K[seed][i]
        ]
        ave_acc_10K[seed] = [
            OFF_10K[seed][i], NONE[seed][i],
            EWC_10K[seed][i],
            MAS_10K[seed][i],
            FLY_10K[seed][i],
            # SDM_10K[seed][i],
            WTK_M_10K[seed][i],
            WTK_S_10K[seed][i],
            WTK_M_EWC_10K[seed][i],
            # SDM_EWC[seed][i],
            WTK_S_EWC_10K[seed][i]
        ]

    act_k_list_2K = {}
    sparse_list_2K = {}
    act_k_list_10K = {}
    sparse_list_10K = {}
    for seed in seed_list:
        act_k_list_2K[seed] = [
            WTK_M_2K[seed][0]['act_k'],
            WTK_S_2K[seed][0]['act_k'],
            WTK_M_EWC_2K[seed][0]['act_k'],
            WTK_S_EWC_2K[seed][0]['act_k'],
        ]
        sparse_list_2K[seed] = [
            WTK_M_2K[seed][0]['sparseness'],
            WTK_S_2K[seed][0]['sparseness'],
            WTK_M_EWC_2K[seed][0]['sparseness'],
            WTK_S_EWC_2K[seed][0]['sparseness'],
        ]
        act_k_list_10K[seed] = [
            WTK_M_10K[seed][0]['act_k'],
            WTK_S_10K[seed][0]['act_k'],
            WTK_M_EWC_10K[seed][0]['act_k'],
            WTK_S_EWC_10K[seed][0]['act_k'],
        ]
        sparse_list_10K[seed] = [
            WTK_M_10K[seed][0]['sparseness'],
            WTK_S_10K[seed][0]['sparseness'],
            WTK_M_EWC_10K[seed][0]['sparseness'],
            WTK_S_EWC_10K[seed][0]['sparseness'],
        ]

    ###########################################################
    # PLOTTING ########

    # name for plot
    plot_name = "summary-{}{}-{}-2K and 10K".format(args.experiment, args.tasks, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    ylabel_all = "Average accuracy (after all tasks)"
    ylabel = "Average accuracy (on tasks seen so far)"
    x_axes = OFF_2K[args.seed][0]['x_task']
    # x_axes = OFF[args.seed][0]["x_task"]

    names = ["Joint", "None", "EWC", "MAS", "FlyModel", 'WTK_M', 'WTK_S', "WTK_M_EWC", 'WTK_S_EWC']
    colors = ["black", "grey", "#CC79A7", "goldenrod", "#56B4E9", 'blue', 'goldenrod', 'purple', 'red']
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    # open pdf
    pp = plotting.open_pdf(f"{args.plot_dir}/{plot_name}.pdf")
    figure_list = []

    ###########################################################
    # bar-plot 2K
    means = [np.mean([ave_acc_2K[seed][id] for seed in seed_list]) for id in ids]
    if args.n_seeds>1:
        sems = [np.sqrt(np.var([ave_acc_2K[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = plotting.plot_bar(means, names=names, colors=colors, ylabel="Test accuracy (after all 10 classes)", title=title,
                               yerr=sems if args.n_seeds > 1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY 2K RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:30s} {:5.2f}  (+/- {:4.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:34s} {:5.2f}".format(name, 100*means[i]))
    print("#"*60)

    ###########################################################
    # line-plot 2K
    ave_lines = []
    sem_lines = []
    for id in ids:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(acc_2K[args.seed][id])):
            all_entries = [acc_2K[seed][id][line_id] for seed in seed_list]
            new_ave_line.append(np.mean(all_entries))
            if args.n_seeds>1:
                new_sem_line.append(np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
        ave_lines.append(new_ave_line)
        sem_lines.append(new_sem_line)
    ylim = (0, 1) if args.scenario=="class" else None
    figure = plotting.plot_lines(ave_lines,
                                x_axes=None,
                                # x_axes=x_axes,
                                line_names=names, colors=colors, title=title,
                                xlabel="# {} so far".format("classes" if args.scenario=="class" else "tasks"),
                                ylabel="Test accuracy (on tasks seen so far)",
                                list_with_errors=sem_lines if args.n_seeds>1 else None, ylim=ylim)
    figure_list.append(figure)

    ###########################################################
    # bar-plot 10K
    means = [np.mean([ave_acc_10K[seed][id] for seed in seed_list]) for id in ids]
    if args.n_seeds>1:
        sems = [np.sqrt(np.var([ave_acc_10K[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = plotting.plot_bar(means, names=names, colors=colors, ylabel="Test accuracy (after all 10 classes)", title=title,
                               yerr=sems if args.n_seeds > 1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY 10K RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:30s} {:5.2f}  (+/- {:4.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:34s} {:5.2f}".format(name, 100*means[i]))
    print("#"*60)

    ###########################################################
    # line-plot 10K
    ave_lines = []
    sem_lines = []
    for id in ids:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(acc_10K[args.seed][id])):
            all_entries = [acc_10K[seed][id][line_id] for seed in seed_list]
            new_ave_line.append(np.mean(all_entries))
            if args.n_seeds > 1:
                new_sem_line.append(np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
        ave_lines.append(new_ave_line)
        sem_lines.append(new_sem_line)
    ylim = (0, 1) if args.scenario=="class" else None
    figure = plotting.plot_lines(ave_lines,
                                x_axes=None,
                                # x_axes=x_axes,
                                line_names=names, colors=colors, title=title,
                                xlabel="# {} so far".format("classes" if args.scenario=="class" else "tasks"),
                                ylabel="Test accuracy (on tasks seen so far)",
                                list_with_errors=sem_lines if args.n_seeds>1 else None, ylim=ylim)
    figure_list.append(figure)

    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print(f"\nGenerated plot: {args.plot_dir}/{plot_name}.pdf\n")

    # get the average sparseness and the act_k ##############################
    methods_len = len(act_k_list_2K[seed_list[0]])
    means_actk_2K = [np.mean([act_k_list_2K[seed][id] for seed in seed_list]) for id in range(methods_len)]
    means_sparseness_2K = [np.mean([sparse_list_2K[seed][id] for seed in seed_list]) for id in range(methods_len)]
    means_actk_10K = [np.mean([act_k_list_10K[seed][id] for seed in seed_list]) for id in range(methods_len)]
    means_sparseness_10K = [np.mean([sparse_list_10K[seed][id] for seed in seed_list]) for id in range(methods_len)]
    if args.n_seeds > 1:
        sems_actk_2K = [np.sqrt(np.var([act_k_list_2K[seed][id] for seed in seed_list]) / (len(seed_list) - 1))
                        for id in range(methods_len)]
        sems_sparse_2K = [np.sqrt(np.var([sparse_list_2K[seed][id] for seed in seed_list]) / (len(seed_list) - 1))
                          for id in range(methods_len)]
        sems_actk_10K = [np.sqrt(np.var([act_k_list_10K[seed][id] for seed in seed_list]) / (len(seed_list) - 1))
                         for id in range(methods_len)]
        sems_sparse_10K = [np.sqrt(np.var([sparse_list_10K[seed][id] for seed in seed_list]) / (len(seed_list) - 1))
                           for id in range(methods_len)]
    act_names = ['WTK_M', 'WTK_S', "WTK_M_EWC", 'WTK_S_EWC']
    for i, name in enumerate(act_names):
        if len(seed_list) > 1:
            print(
                "{:30s} act_num {:5.2f}  (+/- {:4.2f}), n={}".format(name+"_2K", means_actk_2K[i], sems_actk_2K[i], len(seed_list)))
            print("{:30s} sparseness {:5.2f}  (+/- {:4.2f}), n={}".format(name+"_2K", means_sparseness_2K[i], sems_sparse_2K[i],
                                                                             len(seed_list)))
            print(
                "{:30s} act_num {:5.2f}  (+/- {:4.2f}), n={}".format(name+"_10K", means_actk_10K[i], sems_actk_10K[i], len(seed_list)))
            print("{:30s} sparseness {:5.2f}  (+/- {:4.2f}), n={}".format(name+"_10K", means_sparseness_10K[i], sems_sparse_10K[i],
                                                                              len(seed_list)))
        else:
            print("{:30s} act_num {:5.2f}".format(name+"_2K", means_actk_2K[i]))
            print("{:30s} sparseness {:5.2f}".format(name+"_2K", means_sparseness_2K[i]))
            print("{:30s} act_num {:5.2f}".format(name+"_10K", means_actk_10K[i]))
            print("{:30s} sparseness {:5.2f}".format(name+"_10K", means_sparseness_10K[i]))

    print("#" * 60)

