import os
import numpy as np
import torch

from utils.param_stamps import get_param_stamp_from_args
from utils import options, utils
from visual import plotting
import matplotlib.pyplot as plt
import main_exp


adap_param_list = [40000, 100000, 1000000, 2000000, 4000000]
thresh_list = [1.5, 2., 3., 4.]


def fixed_format(list):
    for idx, item in enumerate(list):
        if isinstance(item, torch.Tensor):
            list[idx] = item.cpu().numpy()


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


if __name__ == "__main__":

    args = options.handle_inputs(
        single_task=False, not_include=['perm'], filename='_compare_hyperparams',
        description='Compare performance of different hyper parameters for step mask'
    )

    args.experiment = "CIFAR10_Embeddings_0"
    args.scenario = 'class'
    args.replay = 'none'
    args.use_ann = False
    args.topk = True
    args.use_mask = True
    args.step_mask = True
    args.iters = 20000
    args.adap_param = 2000000

    args.surrogate = 'ATAN'
    args.fc_depth = 2
    args.fc_units = 1000
    args.h_dim = 1000
    args.lr = 5e-2
    args.optimiser = 'sgd'

    args.log_per_task = False
    args.acc_n = None
    args.seed = 12
    args.n_seeds = 3
    args.loss_log = 20000
    args.acc_log = 2000
    args = options.preprocess_args(args, single_task=False)

    # If needed, create plotting directory
    if not os.path.isdir(args.plot_dir):
        os.mkdir(args.plot_dir)

    #########################################################
    # -> ALL CONDITION ##############################

    #########################################################
    # no adaptive thresh ##############
    args.step_mask = True
    args.norm_const = 0.25
    args.k_min = 10
    args.use_mask = True
    args.adaptive_thresh = False
    NO_ADAP = get_results(args)

    args.adaptive_thresh = True

    # SNN_mask #################################################
    args.use_mask = True
    args.topk = True
    args.norm_const = 1.
    args.step_mask = False
    args.use_ann = False
    args.adap_param = 2000000
    args.adaptive_thresh = True
    MASK = get_results(args)

    args.use_mask = True

    # SNN_mask #################################################
    args.use_mask = True
    args.topk = True
    args.step_mask = False
    args.use_ann = False
    args.adap_param = 2000000
    args.adaptive_thresh = True
    MASK_AD = get_results(args)

    args.use_mask = True

    #########################################################
    # hyper parameters in thresh param ##############
    args.step_mask = True
    args.norm_const = 0.25
    args.k_min = 10
    args.use_mask = True
    HYPER = {}
    for choosed_param in adap_param_list:
        args.adap_param = choosed_param
        HYPER[choosed_param] = get_results(args)

    args.adap_param = 2000000

    #########################################################
    # hyper parameters in max_thresh ##############
    args.step_mask = True
    args.norm_const = 0.25
    args.k_min = 10
    args.use_mask = True
    THRES = {}
    for max_thresh in thresh_list:
        args.thresh_max = max_thresh
        THRES[max_thresh] = get_results(args)

    args.thresh_max = 2.

    ###########################################################
    # COLLECT RESULTS ########
    STEP = 5
    # concerning about the lines
    # acc and ave-acc
    ind = 0  # -> the dict
    acc = [NO_ADAP[ind]['average']] + [HYPER[choosed_param][ind]['average'] for choosed_param in adap_param_list]
    acc_max_thresh = [NO_ADAP[ind]['average']] + [THRES[max_th][ind]['average'] for max_th in thresh_list]
    print(1)
    # dead-line
    dead = [MASK[ind]['dead'][::STEP]] + [NO_ADAP[ind]['dead'][::STEP]] + \
           [HYPER[choosed_param][ind]['dead'][::STEP] for choosed_param in adap_param_list]

    # sparse-line
    sparse = [MASK[ind]['sparseness'][::STEP]] + [NO_ADAP[ind]['sparseness'][::STEP]] + \
             [HYPER[choosed_param][ind]['sparseness'][::STEP] for choosed_param in adap_param_list]

    act_k = [MASK[ind]['act_k'][::STEP]] + [NO_ADAP[ind]['act_k'][::STEP]] + \
            [HYPER[choosed_param][ind]['act_k'][::STEP] for choosed_param in adap_param_list]

    for dead_lines in dead:
        fixed_format(dead_lines)
    for sparse_line in sparse:
        fixed_format(sparse_line)
    for sparse_line in sparse:
        fixed_format(sparse_line)
    # concerning about the values
    ind = 1  # -> the average acc
    ave_acc = [NO_ADAP[ind]] + [HYPER[choosed_param][ind] for choosed_param in adap_param_list]
    ave_acc_th = [NO_ADAP[ind]] + [THRES[max_th][ind] for max_th in thresh_list]

    ###########################################################
    # PLOTTING ########
    plot_name = f"hyperParam-Test-{args.experiment}{args.tasks}-{args.scenario}"
    scheme = f"incremental {args.scenario} learning"
    title = f"{args.experiment} - {scheme}"
    x_axes = NO_ADAP[0]["x_iteration"]

    colors = ['#F7969E', '#86C7B8', '#F6D18A', '#BF4040', '#5F9ED1', '#8EBA42', '#FFD966', '#A6D8DE', '#D8BFD8', '#D0D0D0']

    names_mask = ["Mask"] + ["NoAdapt"] + [f"param={choosed_param}" for choosed_param in adap_param_list]
    colors_mask = ['black', 'grey'] + [one_color for one_color in colors[:len(adap_param_list)]]
    names_normal = ["NoAdapt"] + [f"param={choosed_param}" for choosed_param in adap_param_list]
    colors_normal = ['grey'] + [one_color for one_color in colors[:len(adap_param_list)]]
    names_thres = ["NoAdapt"] + [f"max_th={max_th}" for max_th in thresh_list]
    colors_thres = ['grey'] + [one_color for one_color in colors[-len(adap_param_list):]]

    pp = plotting.open_pdf(f"{args.plot_dir}/{plot_name}.pdf")
    figure_list = []

    ######################################################
    # bar-plot
    one_title = "Adaptive threshold Parameter tune"
    figure = plotting.plot_bar(numbers=ave_acc, names=names_normal, colors=colors_normal, ylim=[0, 1], title=one_title)
    figure_list.append(figure)

    for i, name in enumerate(names_normal):
        print("{:19s} {:.2f}".format(name, 100*ave_acc[i]))
    print("#"*60)

    one_title = "Max threshold Parameter tune"
    figure = plotting.plot_bar(numbers=ave_acc_th, names=names_thres, colors=colors_thres, ylim=[0, 1], title=one_title)
    figure_list.append(figure)

    for i, name in enumerate(names_thres):
        print("{:19s} {:.2f}".format(name, 100 * ave_acc_th[i]))
    print("#" * 60)

    # line-plot
    one_title = "Adaptive threshold Parameter tune acc-line"
    figure = plotting.plot_lines(
        list_with_lines=acc, x_axes=x_axes,
        line_names=names_normal, colors=colors_normal, title=one_title,
    )
    figure_list.append(figure)

    one_title = "Adaptive threshold Parameter tune dead-line"
    figure = plotting.plot_lines(
        list_with_lines=dead, x_axes=x_axes[::STEP],
        line_names=names_mask, colors=colors_mask, title=one_title,
    )
    figure_list.append(figure)

    one_title = "Adaptive threshold Parameter tune sparse-line"
    figure = plotting.plot_lines(
        list_with_lines=sparse, x_axes=x_axes[::STEP],
        line_names=names_mask, colors=colors_mask, title=one_title
    )
    figure_list.append(figure)

    for figure in figure_list:
        pp.savefig(figure)

    pp.close()
    print(f"\nGenerated plot: {args.plot_dir}/{plot_name}.pdf\n")




