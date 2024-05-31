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
        single_task=False, not_include=["perm"], filename="_compare_MNIST",
        description="Compare performance of continual learning strategies on different scenarios of split MNIST."
    )

    ################################################################################
    # Add default arguments (will be different for different runs)
    args.experiment = "splitMNIST"
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
    args.iters = 10000
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
    args.h_dim = 1000
    args.fc_units = 1000
    args.lr = 5e-2
    args.optimiser = 'sgd'

    args.log_per_task = False
    args.acc_n = None
    args.loss_log = 10000
    args.acc_log = 1000
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
    # Baseline ########
    print(args.scenario)
    # Joint
    args.replay = "offline"
    args.topk = False

    OFF = {}
    OFF = collect_all(OFF, seed_list, args, name="Joint")

    # None
    args.replay = "none"
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")

    args.replay = "none"

    ###########################################################
    # Other methods ########
    args.topk = True
    args.use_mask = True
    args.hard_mask = False
    args.use_ann = False
    ###########################################################
    # FlyModel ################################################
    ###########################################################
    args.model_plat = 'flymodel'
    args.iters = 100
    args.acc_log = 10
    args.n_kc = 1000
    args.k_num = 10
    args.lr = 0.005
    args.min_max = True
    args.kc_response = 64
    args.record_process = False

    FLY_1K = {}
    FLY_1K = collect_all(FLY_1K, seed_list, args, name="FLY_1K")

    # args.n_kc = 10000
    # args.k_num = 100
    # FLY_10K = {}
    # FLY_10K = collect_all(FLY_10K, seed_list, args, name="FLY_10K")

    args.model_plat = 'standard'
    args.iters = 10000
    args.acc_log = 1000
    args.lr = 5e-2

    # ANN_SDM #################################################
    args.model_plat = 'standard'
    args.k_approach = "GABA_SWITCH"
    args.use_mask = False
    args.topk = True
    args.use_ann = True
    args.gaba_switch_num = 800000
    SDM = {}
    SDM = collect_all(SDM, seed_list, args, name="ANN_SDM")

    args.use_mask = True
    args.use_ann = False
    args.iters = 10000

    # ANN_SDM + EWC ###########################################
    args.model_plat = 'standard'
    args.k_approach = "GABA_SWITCH"
    args.use_mask = False
    args.topk = True
    args.use_ann = True
    args.ewc = True
    args.simpled_EWC = True
    args.ewc_lambda = 800
    args.ewc_beta = 0.005
    args.gaba_switch_num = 800000
    SDM_EWC = {}
    SDM_EWC = collect_all(SDM_EWC, seed_list, args, name="SDM_EWC")

    args.use_mask = True
    args.use_ann = False
    args.k_approach = "FLAT"
    args.ewc = False
    #
    # SNN_mask #################################################
    args.use_mask = True
    args.topk = True
    args.step_mask = False
    args.use_ann = False
    args.adap_param = 1500000
    WTK_M = {}
    WTK_M = collect_all(WTK_M, seed_list, args, name="WTK_M")

    args.use_mask = True

    # SNN_mask + EWC ##########################################
    args.use_mask = True
    args.topk = True
    args.step_mask = False
    args.use_ann = False
    args.adap_param = 1500000
    args.ewc = True
    args.simpled_EWC = True
    args.ewc_lambda = 400
    args.ewc_beta = 0.08
    WTK_M_EWC = {}
    WTK_M_EWC = collect_all(WTK_M_EWC, seed_list, args, name="WTK_M_EWC")

    args.use_mask = True
    args.ewc = False

    # SNN_step ###########################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True
    args.norm_const = 1.
    args.adap_param = 1000000
    WTK_S = {}
    WTK_S = collect_all(WTK_S, seed_list, args, name="WTK_S")

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.

    # SNN_step + EWC #####################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True
    args.adap_param = 1000000
    args.norm_const = 1.
    args.ewc = True
    args.simpled_EWC = True
    args.ewc_lambda = 400
    args.ewc_beta = 0.08
    WTK_S_EWC = {}
    WTK_S_EWC = collect_all(WTK_S_EWC, seed_list, args, name="WTK_S_EWC")

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.
    args.ewc = False

    # # SNN_step_2 ###########################################
    # args.use_mask = True
    # args.topk = True
    # args.use_ann = False
    # args.step_mask = True
    # args.norm_const = 1.
    # args.adap_param = 1000000
    # args.tr_decay = -10000
    # WTK_S_2 = {}
    # WTK_S_2 = collect_all(WTK_S_2, seed_list, args, name="WTK_S_2")
    #
    # args.use_mask = True
    # args.use_ann = False
    # args.norm_const = 1.
    # args.tr_decay = 10
    #
    # # SNN_step_2 + EWC #####################################
    # args.use_mask = True
    # args.topk = True
    # args.use_ann = False
    # args.step_mask = True
    # args.adap_param = 1000000
    # args.norm_const = 1.
    # args.ewc = True
    # args.simpled_EWC = True
    # args.tr_decay = -10000
    # args.ewc_lambda = 400
    # args.ewc_beta = 0.08
    # WTK_S_EWC_2 = {}
    # WTK_S_EWC_2 = collect_all(WTK_S_EWC_2, seed_list, args, name="WTK_S_EWC_2")
    #
    # args.use_mask = True
    # args.use_ann = False
    # args.norm_const = 1.
    # args.tr_decay = 10
    # args.ewc = False

    # EWC
    args.ewc = True
    args.simple_EWC = True
    args.ewc_lambda = 800
    args.ewc_beta = 0.32
    args.topk = False
    args.use_ann = False
    EWC = {}
    EWC = collect_all(EWC, seed_list, args, name="EWC")
    args.ewc = False
    args.topk = True

    # MAS
    args.mas = True
    args.mas_lambda = 10000.
    args.topk = False
    args.use_ann = False
    MAS = {}
    MAS = collect_all(MAS, seed_list, args, name="MAS")
    args.mas = False
    args.topk = True

    ###########################################################
    # COLLECT RESULTS ########
    acc = {}
    ave_acc = {}
    task_acc = {}

    task_num = args.tasks
    # Create lists for all extracted <dicts> and <lists> with fixed order
    for seed in seed_list:

        i = 0
        acc[seed] = [
            OFF[seed][i]["average"], NONE[seed][i]["average"],
            EWC[seed][i]['average'],
            MAS[seed][i]['average'],
            FLY_1K[seed][i]["average"],
            SDM[seed][i]["average"],
            WTK_M[seed][i]["average"],
            # WTK_M_EWC[seed][i]["average"],
            WTK_S[seed][i]["average"],
            SDM_EWC[seed][i]["average"],
            WTK_S_EWC[seed][i]["average"],
            # WTK_S_2[seed][i]["average"], WTK_S_EWC_2[seed][i]["average"],
        ]
        add_zeroInacc(WTK_S, seed_list, task_num=task_num)
        task_acc[seed] = [
            WTK_S_EWC[seed][i]["all_tasks"][task_id] for task_id in range(task_num)
        ]
        i = 1
        ave_acc[seed] = [
            OFF[seed][i], NONE[seed][i],
            EWC[seed][i],
            MAS[seed][i],
            FLY_1K[seed][i],
            SDM[seed][i],
            WTK_M[seed][i],
            # WTK_M_EWC[seed][i],
            WTK_S[seed][i],
            SDM_EWC[seed][i],
            WTK_S_EWC[seed][i],
            # WTK_S_2[seed][i], WTK_S_EWC_2[seed][i],
        ]

    ###########################################################
    # PLOTTING ########

    # name for plot
    plot_name = "summary-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    scheme = "incremental {} learning".format(args.scenario)
    title = "{}  -  {}".format(args.experiment, scheme)
    ylabel_all = "Average accuracy (after all tasks)"
    ylabel = "Average accuracy (on tasks seen so far)"
    x_axes = SDM[args.seed][0]['x_task']
    # x_axes = OFF[args.seed][0]["x_task"]

    # select names / colors / ids
    # names = ["None", "LwF", "EWC", "SI", "Generative Replay (GR)", "Brain-Inspired Replay (BI-R)",
    #          "BI-R + SI" if args.scenario == "class" else "Joint"]
    # colors = ["grey", "goldenrod", "darkgreen", "yellowgreen", "red", "purple", "blue" if args.scenario=="class" else "black"]
    # ids = [1, 2, 4, 5, 3, 6, 7 if args.scenario == "class" else 0]
    names = [
        "Joint", "None",
        "EWC", "MAS",
        "FlyModel", "SDM",
        "WTK_M", "WTK_S",
        "SDM_EWC", "WTK_S_EWC",
        ]
    colors = [
        "black", "grey",
        "#CC79A7", "goldenrod",
        "#56B4E9", 'green',
        'blue', 'goldenrod',
        'yellowgreen', 'red']
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # open pdf
    pp = plotting.open_pdf(f"{args.plot_dir}/{plot_name}.pdf")
    figure_list = []

    # bar-plot
    means = [np.mean([ave_acc[seed][id] for seed in seed_list]) for id in ids]
    if args.n_seeds>1:
        sems = [np.sqrt(np.var([ave_acc[seed][id] for seed in seed_list])/(len(seed_list)-1)) for id in ids]
    figure = plotting.plot_bar(means, names=names, colors=colors, ylabel="Test accuracy (after all 10 classes)", title=title,
                               yerr=sems if args.n_seeds>1 else None, ylim=(0,1))
    figure_list.append(figure)

    # print results to screen
    print("\n\n"+"#"*60+"\nSUMMARY RESULTS: {}\n".format(title)+"-"*60)
    for i,name in enumerate(names):
        if len(seed_list) > 1:
            print("{:30s} {:5.2f}  (+/- {:4.2f}),  n={}".format(name, 100*means[i], 100*sems[i], len(seed_list)))
        else:
            print("{:34s} {:5.2f}".format(name, 100*means[i]))
    print("#"*60)

    # line-plot
    ave_lines = []
    sem_lines = []
    for id in ids:
        new_ave_line = []
        new_sem_line = []
        for line_id in range(len(acc[args.seed][id])):
            all_entries = [acc[seed][id][line_id] for seed in seed_list]
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

    # appended all tasks
    ave_tasks_lines = []
    sem_tasks_lines = []
    for task_id in range(task_num):
        new_tasks_acc = []
        new_tasks_sem = []
        for point_id in range(len(task_acc[args.seed][task_id])):
            all_entries = [task_acc[seed][task_id][point_id] for seed in seed_list]
            new_tasks_acc.append(np.mean(all_entries))
            if args.n_seeds > 1:
                new_tasks_sem.append(np.sqrt(np.var(all_entries)/(len(all_entries)-1)))
        ave_tasks_lines.append(new_tasks_acc)
        sem_tasks_lines.append(new_tasks_sem)
    Task_names = ['T1', 'T2', 'T3', 'T4', 'T5']
    Task_colors = ["#E74C3C", "#F39C12", "#FFEB3B", "#4CAF50", "#1ABC9C"]
    figure = plotting.plot_lines(
        ave_tasks_lines, x_axes=None, line_names=Task_names, colors=Task_colors,
        title="ave Task acc curves", list_with_errors=sem_tasks_lines if args.n_seeds > 1 else None, ylim=ylim
    )
    figure_list.append(figure)

    figure_simple = plotting.plot_lines(
        task_acc[seed_list[0]], x_axes=None, line_names=Task_names, colors=Task_colors,
        title="Task acc curves", list_with_errors=None, ylim=ylim
    )
    figure_list.append(figure_simple)

    # add figures to pdf
    for figure in figure_list:
        pp.savefig(figure)

    # close the pdf
    pp.close()

    # Print name of generated plot on screen
    print(f"\nGenerated plot: {args.plot_dir}/{plot_name}.pdf\n")