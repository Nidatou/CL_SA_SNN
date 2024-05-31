import argparse
import os
import numpy as np
import torch

from data_utils.utils import get_multitask_experiment
from models.cl.continual_learner import CLBase
from models.fc.nets import MLP_attention
from models.FlyModel import FlyModel
from models.FlyModel_whole import FlyModel_Whole
from models.simple_MLP import Simple
from models.snn_kWTA import Gate_snn
from train import train_cl
from evaluations import evaluate, callbacks
from utils import options, utils
from utils import param_stamps
from utils.model_definition import define_SDM_classifier, define_Snn_WTAk, define_specific_model


def run(args: argparse.Namespace, verbose=False):
    #########################################################
    # pre process for training ##############################
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir(args.plot_dir):
        os.mkdir(args.plot_dir)

    cuda = torch.cuda.is_available()
    device = torch.device(args.device if cuda else "cpu")
    args.device = device
    if verbose:
        print(f"The model deployed on {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # modify the collate
    if args.experiment in ['splitNMNIST', 'NMNIST']:
        if args.use_ann is True or args.model_plat not in ['standard']:
            utils.collate_func = utils.rewind_collate_fn
        else:
            utils.collate_func = utils.spiking_collate_fn

    #########################################################
    # Data preparation #################################
    if verbose:
        print("\nPreparing the data...")
    (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=args.tasks, data_dir=args.data_dir,
        normalize=not args.no_norm, augment=args.augment,
        verbose=verbose, exception=True if args.seed < 10 else False,
        only_test=not args.train, T=args.T,
    )

    #########################################################
    # Main model ######################################
    if verbose:
        print('\n Defining the model...')

    model = define_specific_model(args, config, device=device)
    model = utils.init_params(model, args)
    # check the weights of conv-layers?
    if utils.checkattr(args, 'freeze_convE'):
        for param in model.convE.parameters():
            param.requires_grad = False
        if verbose:
            print('\n Encoder conv layer fixed!')
    if utils.checkattr(args, 'freeze_atten') and isinstance(model.fcPart, MLP_attention):
        model.fcPart.freeze_atten()
        if verbose:
            print('\n Attention gate have been fixed')

    #########################################################
    # Strategy(for regularization/replay) ###################
    model.grad_clip = args.grad_clip
    # Strategy for regularization #######
    if isinstance(model, CLBase) and utils.checkattr(args, 'ewc'):
        model.ewc_lambda = args.ewc_lambda if args.ewc else 0
        model.fisher_n = args.fisher_n
        model.online = utils.checkattr(args, 'online')
        if model.online:
            model.gamma = args.gamma

        model.simpled_EWC = args.simpled_EWC
        if model.simpled_EWC:
            model.ewc_beta = args.ewc_beta
            model.cl_batches = args.cl_batches
            model.cl_batch_size = args.cl_batch_size

    if isinstance(model, CLBase) and args.mas:
        model.mas_lambda = args.mas_lambda if args.mas else 0
        model.online_mas = args.online_mas
        model.cl_batches = args.cl_batches
        model.cl_batch_size = args.cl_batch_size

    # Synaptic Intelligence
    if isinstance(model, CLBase) and utils.checkattr(args, 'si'):
        model.si_c = args.si_c if args.si else 0
        model.epsilon = args.epsilon

    # XdG
    if isinstance(model, CLBase) and utils.checkattr(args, 'xdg') and args.xdg_prop > 0:
        model.define_XdGmask(gating_prop=args.xdg_prop, n_tasks=args.tasks)

    # Strategy for regularization #######
    if isinstance(model, CLBase) and hasattr(args, 'replay') and not args.replay == 'none':
        model.replay_targets = "soft" if args.distill else 'hard'
        model.KD_temp = args.temp

    generator = None

    #########################################################
    # Evaluation setting ####################################
    if verbose:
        print('\n parameter-stamp...')
    param_stamp = param_stamps.get_param_stamp(
        args, model.name, verbose,
        replay=True if (hasattr(args, 'replay') and not args.replay == "none") else False,
    )
    # record all the accuracy of tasks
    progress_dict = evaluate.initiate_progress_dict(args.tasks)

    # do something for plotting in visdom
    visdom = None
    # --------------------------------------

    #########################################################
    # Callbacks #####################################
    # the main model(whole classifier callbacks)
    solver_loss_cbs = [
        callbacks._solver_loss_cb(log_interval=args.loss_log, visdom=visdom, model=model, iters_per_task=args.iters,
                                  tasks=args.tasks, replay=False)
    ]

    process_cbs = [
        callbacks._process_cb(log_interval=args.iters, test_datasets=test_datasets, progress_dict=progress_dict,
                              iters_per_task=args.iters, test_size=None, classes_per_task=classes_per_task,
                              scenario=args.scenario, mid_size=args.h_dim)
    ] if args.record_process else []

    # callback for reporting and visualizing accuracy
    eval_cb = callbacks._eval_cb(
        log_interval=args.acc_log, test_datasets=test_datasets, visdom=visdom, progress_dict=None,
        iters_per_task=args.iters, test_size=args.acc_n, classes_per_task=classes_per_task, scenario=args.scenario,
        verbose=verbose,
    )
    eval_cb_full = callbacks._eval_cb(
        log_interval=args.acc_log, test_datasets=test_datasets, visdom=visdom, progress_dict=progress_dict,
        iters_per_task=args.iters, classes_per_task=classes_per_task, scenario=args.scenario,
        verbose=verbose, 
    )
    eval_cbs = [eval_cb, eval_cb_full]

    #########################################################
    # Training ######################################
    g_iters = args.g_iters if hasattr(args, 'g_iters') else args.iters
    if args.train:
        if verbose:
            print('\nTraining...')
        train_cl(model, train_datasets, replay_mode=args.replay if hasattr(args, 'replay') else 'none',
                 scenario=args.scenario, classes_per_task=classes_per_task, iters=args.iters,
                 iters_first=args.iters if args.iters_first is None else args.iters_first,
                 batch_size=args.batch, batch_size_replay=args.batch_replay if hasattr(args, 'batch_replay') else None,
                 generator=None, gen_iters=args.iters, feedback=utils.checkattr(args, 'feedback'),
                 sample_in_model=False, args=args,
                 eval_cbs=eval_cbs, gen_loss_cbs=None, loss_cbs=solver_loss_cbs, process_cbs=process_cbs)
        file_name = f"{args.results_dir}/dict-{param_stamp}"
        utils.save_object(progress_dict, file_name)
        if args.save:
            save_name = f"mM-{param_stamp}"
            utils.save_checkpoint(model, args.model_dir, args=args, name=save_name, verbose=verbose)

    else:
        if verbose:
            print('\nLoading parameters of the previously trained models...')
        load_name = f"mM-{param_stamp}"
        utils.load_checkpoint(model, args.model_dir, name=load_name, verbose=verbose)

        # load 和 save 都有跟generator有关的部分

    #########################################################
    # Evaluation of classifier ##############################
    if verbose:
        print('\n EVALUATION RESULTS:')

    # evaluare accuracy of final model on full test-set
    accs = [evaluate.validate(
        model, test_datasets[i], verbose=False, test_size=None, task=i+1,
        allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if args.scenario == 'task' else None
    ) for i in range(args.tasks)]
    average_accs = sum(accs)/args.tasks
    if verbose:
        print('\n Accuracy of final model on test-set:')
        for i in range(args.tasks):
            print(f"- {'For classes from task' if args.scenario == 'class' else 'Task'} {i+1}: {accs[i]:.4f}")
        print(f'=> Average accuracy over all {args.tasks * classes_per_task if args.scenario=="class" else args.tasks} '
              f'{"classes" if args.scenario=="class" else "tasks"}: {average_accs:.4f}\n')
    output_file = open(f"{args.results_dir}/acc-{param_stamp}.txt", 'w')
    output_file.write(f'{average_accs}\n')
    output_file.close()

    #########################################################
    # Then is the evaluation of generator ##################

    # and plotting #########################################
    # -----------------------------------


if __name__ == "__main__":
    # args = handle_inputs()
    args = options.handle_inputs(False, filename='test', not_include=['perm'],
                                 description='Compare & combine continual learning approaches')
    args.scenario = 'class'
    args.experiment = "CIFAR100_Embeddings"
    args.fc_depth = 2
    args.h_dim = 2000
    args.norm_const = 0.5
    args.step_mask = True
    args.use_ann = False
    args.use_mask = True
    args.loosen_tr = True
    args.lr = 5e-2
    args = options.preprocess_args(args, single_task=False)
    run(args, verbose=True)

