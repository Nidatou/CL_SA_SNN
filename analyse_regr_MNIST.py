from utils.param_stamps import get_param_stamp_from_args
from utils import options, utils, model_definition, param_stamps
from data_utils.utils import get_singletask_experiment, get_multitask_experiment
from visual import plotting
import main_exp

import numpy as np
import torch
from torch.utils.data import ConcatDataset
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from evaluations import evaluate

import os

import statsmodels.api as sm
from scipy.stats import zscore


def get_confusion_matrix(args, seed, save_root, type_name):
    fig_name = type_name + f'_confusion_s{seed}_mnist.png'
    if os.path.isfile(os.path.join(save_root, fig_name)):
        print(f"{fig_name}: already exist")
        return

    temp_seed = args.seed
    args.seed = seed
    args.train = False
    cuda = torch.cuda.is_available()
    device = torch.device(args.device if cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)
    # I have to load the multi-task dataset first for the chaotic rank of the output
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=args.tasks, data_dir=args.data_dir,
        normalize=not args.no_norm, augment=args.augment,
        verbose=False, exception=True if args.seed < 10 else False,
        only_test=not args.train, T=args.T,
    )

    model = model_definition.define_specific_model(args, config, device=device)
    param_stamp = param_stamps.get_param_stamp(
        args, model.name, verbose=False,
        replay=True if (hasattr(args, 'replay') and not args.replay == "none") else False,
    )
    load_name = f"mM-{param_stamp}"
    utils.load_checkpoint(model, args.model_dir, name=load_name, verbose=True)

    test_set = ConcatDataset(test_datasets[:])
    model.eval()
    test_loader = utils.get_data_loader(test_set, batch_size=args.batch, cuda=cuda, shuffle=False)
    y_list = []
    prediced_list = []
    for batch_idx, (data, y) in enumerate(test_loader):
        data, y = data.to(device), y.to(device)
        scores = model.classify(data)
        _, predicted = torch.max(scores, dim=1)
        y_list.append(y.cpu().numpy().ravel())
        prediced_list.append(predicted.cpu().numpy().ravel())
    y_list = np.concatenate(y_list)
    predicted_list = np.concatenate(prediced_list)
    fig, axs = plt.subplots(nrows=1, ncols=1)
    ConfusionMatrixDisplay.from_predictions(y_true=y_list, y_pred=predicted_list, labels=None, display_labels=None,
                                            cmap=plt.cm.coolwarm, ax=axs)
    axs.set_title('Confusion Matrix')
    axs.set_xlabel('Predicted Label')
    axs.set_ylabel('True Label')
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    fig.tight_layout(pad=1.5)

    plt.savefig(os.path.join(save_root, fig_name), dpi=300)
    print(f"Confusion matrix {fig_name} saved")
    # plt.show()

    args.seed = temp_seed
    args.train = True


# return <progress_dict, ave>
def get_dicts(args):
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run
    assert os.path.isfile(f"{args.results_dir}/dict-{param_stamp}.pkl")
    file_name = f'{args.results_dir}/acc-{param_stamp}.txt'
    file = open(file_name)
    ave = float(file.readline())
    file.close()

    dict = utils.load_object(f"{args.results_dir}/dict-{param_stamp}")

    return dict, ave


def collect_all(method_dict, seed_list, args, name=None):
    if name is not None:
        print(f'\n------{name}------')
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_dicts(args)

    return method_dict


def save_weight_histogram(method_dict, seed, type_name, save_root, y_upper=1000):
    classes = 10
    wt_list = method_dict[seed][0]['process_weight']
    tasks = len(wt_list)
    classes_per_task = int(np.floor(classes/tasks))

    plt.rcParams.update({'font.size': 13})

    xmin, xmax = -1, 1
    bins = np.linspace(xmin, xmax, 21)
    for step in range(tasks):
        fig, axs = plt.subplots(nrows=1, ncols=1)
        for sub_step in range(step+1):
            axs.hist(
                wt_list[step][sub_step * classes_per_task: (sub_step+1) * classes_per_task].ravel(),
                bins=bins, alpha=0.5, label=f"Task {sub_step}"
            )
        axs.set_title('Histogram of weight matrix')
        axs.set_xlabel('weight magnitude')
        axs.set_ylabel('density')
        axs.set_ylim([0, y_upper])
        axs.legend(loc="upper right")
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)

        fig.tight_layout(pad=1.5)
        fig_name = type_name + f'_{step}.png'
        plt.savefig(os.path.join(save_root, fig_name), dpi=300)
        print(f'save histogram {fig_name}')


def save_spk_histogram(method_dict, seed, type_name, save_root, item_name='total'):
    classes = 10
    assert item_name in ['total', 'true']
    key_name = 'all_classes_spk' if item_name == 'total' else 'true_classes_spk'
    spk_list = method_dict[seed][0][key_name]
    tasks = len(spk_list)
    classes_per_task = int(np.floor(classes / tasks))

    plt.rcParams.update({'font.size': 13})

    xmin, xmax = 0, 6
    bins = np.linspace(xmin, xmax, 61)
    for step in range(tasks):
        fig, axs = plt.subplots(nrows=1, ncols=1)
        for sub_step in range(step + 1):
            log_spk_num = np.log10(
                np.sum(spk_list[step][sub_step * classes_per_task: (sub_step + 1) * classes_per_task], axis=0) + 1
            )
            axs.hist(
                log_spk_num,
                bins=bins, alpha=0.5, label=f"Task {sub_step}"
            )
        axs.set_title('Histogram of spike distribute')
        axs.set_xlabel('spike sum(log10)')
        axs.set_ylabel('neuron_num')
        axs.legend(loc="upper right")
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)

        fig.tight_layout(pad=1.5)
        fig_name = type_name + f'_spk_{step}_s{seed}.png'
        plt.savefig(os.path.join(save_root, fig_name), dpi=300)
        print(f'save spk histogram {fig_name}')

        regr_res = test_for_regr(spk_matrix=spk_list[step], total_class=classes)


def save_specific_neuron_distribute(method_dict, seed, type_name, save_root, item_name='total', ylim=None):
    # fig_name = type_name + f"_regr.png"
    fig_name = type_name + f'_regr_s{seed}_mnist.png'
    if os.path.isfile(os.path.join(save_root, fig_name)):
        print(f"{fig_name}: already exist")
        return

    classes = 10
    assert item_name in ['total', 'true']
    key_name = 'all_classes_spk' if item_name == 'total' else 'true_classes_spk'
    spk_list = method_dict[seed][0][key_name]
    tasks = len(spk_list)
    classes_per_task = int(np.floor(classes / tasks))

    plt.rcParams.update({'font.size': 13})
    neu_num_list = []
    for step in range(tasks):
        regr_res = test_for_regr(spk_matrix=spk_list[step], total_class=classes)
        neu_num_list.append(regr_res)
    neu_num_list = np.stack(neu_num_list, axis=1)
    labels = ['T1', 'T2', 'T3', 'T4', 'T5']
    stack_name = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
    plotting.plot_stacked_area_chart(
        labels=labels, stacked_info=neu_num_list, stack_name=stack_name, ylim=ylim)
    plt.savefig(os.path.join(save_root, fig_name), dpi=300)
    print(f'save regr stacked plotting {fig_name}')


def test_for_regr(spk_matrix, total_class=10):
    class_num = spk_matrix.shape[0]
    neu_num = spk_matrix.shape[1]
    selectivity_matrix = np.zeros((neu_num, class_num))
    dmat = np.zeros((class_num, class_num))
    for i in range(class_num):
        dmat[i, i] = 1
    for i_neuron in range(neu_num):
        neu_spk = spk_matrix[:, i_neuron]
        # regr_model = sm.OLS(neu_spk, dmat)
        regr_model = sm.OLS(zscore(neu_spk), zscore(dmat, axis=0, ddof=1))
        regr_results = regr_model.fit()
        if np.sum(regr_results.tvalues > 1.96) == 1:
            selectivity_matrix[
                i_neuron,
                np.where(regr_results.tvalues == np.max(regr_results.tvalues))[0][0],
            ] = 1
    specific = []
    for class_id in range(class_num):
        # zero_mask = [i for i in range(class_num) if i != class_id]
        test_plat = np.zeros((class_num))
        test_plat[class_id] = 1
        spe_task = np.all(selectivity_matrix == test_plat, axis=1)
        # spe_task = (selectivity_matrix[:, class_id] == 1 & selectivity_matrix[:, zero_mask] == 0)
        spe_num = np.sum(spe_task)
        specific.append(spe_num)
    for added_ in range(total_class - class_num):
        specific.append(0)

    return np.array(specific)


if __name__ == '__main__':

    args = options.handle_inputs(
        single_task=False, not_include=["perm"], filename="_compare_CIFAR10",
        description="Compare performance of continual learning strategies on different scenarios of split CIFAR-10."
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
    args.fc_units = 1000
    args.h_dim = 1000
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

    get_confusion_matrix(args, seed=seed_list[0], save_root=args.plot_dir, type_name='joint')

    # None
    args.replay = "none"
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")

    get_confusion_matrix(args, seed=seed_list[0], save_root=args.plot_dir, type_name='none')

    args.replay = "none"

    ###########################################################
    # Other methods ########
    args.topk = True
    args.use_mask = True
    args.hard_mask = False
    args.use_ann = False
    # ###########################################################
    # # FlyModel ################################################
    # ###########################################################
    # args.model_plat = 'flymodel'
    # args.iters = 40
    # args.acc_log = 4
    # args.n_kc = 1000
    # args.k_num = 10
    # args.lr = 0.005
    # args.min_max = True
    # args.kc_response = 32
    #
    # FLY_1K = {}
    # FLY_1K = collect_all(FLY_1K, seed_list, args, name="FLY_1K")
    #
    # # args.n_kc = 10000
    # # args.k_num = 100
    # # FLY_10K = {}
    # # FLY_10K = collect_all(FLY_10K, seed_list, args, name="FLY_10K")
    #
    # args.model_plat = 'standard'
    # args.iters = 20000
    # args.acc_log = 2000
    # args.lr = 5e-2

    # ANN_SDM #################################################
    args.model_plat = 'standard'
    args.k_approach = "GABA_SWITCH"
    args.use_mask = False
    args.topk = True
    args.use_ann = True
    args.gaba_switch_num = 800000
    SDM = {}
    SDM = collect_all(SDM, seed_list, args, name="ANN_SDM")

    get_confusion_matrix(args, seed=seed_list[0], save_root=args.plot_dir, type_name='sdm')

    args.use_mask = True
    args.use_ann = False

    # # ANN_SDM + EWC ###########################################
    # args.model_plat = 'standard'
    # args.k_approach = "GABA_SWITCH"
    # args.use_mask = False
    # args.topk = True
    # args.use_ann = True
    # args.ewc = True
    # args.simpled_EWC = True
    # args.ewc_lambda = 800
    # args.ewc_beta = 0.005
    # args.gaba_switch_num = 800000
    # SDM_EWC = {}
    # SDM_EWC = collect_all(SDM_EWC, seed_list, args, name="SDM_EWC")
    #
    # args.use_mask = True
    # args.use_ann = False
    # args.k_approach = "FLAT"
    # args.ewc = False

    # SNN_mask #################################################
    args.use_mask = True
    args.topk = True
    args.step_mask = False
    args.use_ann = False
    args.adap_param = 1500000
    WTK_M = {}
    WTK_M = collect_all(WTK_M, seed_list, args, name="WTK_M")

    get_confusion_matrix(args, seed=seed_list[0], save_root=args.plot_dir, type_name='snn_mask')

    args.use_mask = True

    # # SNN_mask + EWC ##########################################
    # args.use_mask = True
    # args.topk = True
    # args.step_mask = False
    # args.use_ann = False
    # args.adap_param = 1500000
    # args.ewc = True
    # args.simpled_EWC = True
    # args.ewc_lambda = 400
    # args.ewc_beta = 0.08
    # WTK_M_EWC = {}
    # # WTK_M_EWC = collect_all(WTK_M_EWC, seed_list, args, name="WTK_M_EWC")
    #
    # args.use_mask = True
    # args.ewc = False

    # SNN_step ###########################################
    args.use_mask = True
    args.topk = True
    args.use_ann = False
    args.step_mask = True
    args.norm_const = 1.
    args.adap_param = 1000000
    args.k_min = 10
    WTK_S = {}
    WTK_S = collect_all(WTK_S, seed_list, args, name="WTK_S")

    get_confusion_matrix(args, seed=seed_list[0], save_root=args.plot_dir, type_name='snn_step')

    args.use_mask = True
    args.use_ann = False
    args.norm_const = 1.

    # # SNN_step + EWC #####################################
    # args.use_mask = True
    # args.topk = True
    # args.use_ann = False
    # args.step_mask = True
    # args.adap_param = 1000000
    # args.k_min = 10
    # args.norm_const = 1.
    # args.ewc = True
    # args.simpled_EWC = True
    # args.ewc_lambda = 400
    # args.ewc_beta = 0.08
    # WTK_S_EWC = {}
    # WTK_S_EWC = collect_all(WTK_S_EWC, seed_list, args, name="WTK_S_EWC")
    #
    # args.use_mask = True
    # args.use_ann = False
    # args.norm_const = 1.

    save_specific_neuron_distribute(SDM, seed=seed_list[0], type_name='sdm', save_root=args.plot_dir, ylim=[0, 1000])
    save_specific_neuron_distribute(WTK_M, seed=seed_list[0], type_name='snn_mask', save_root=args.plot_dir, ylim=[0, 1000])
    save_specific_neuron_distribute(WTK_S, seed=seed_list[0], type_name='snn_step', save_root=args.plot_dir, ylim=[0, 1000])
