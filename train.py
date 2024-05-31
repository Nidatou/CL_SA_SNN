import tqdm
import copy
import cv2
import numpy as np
import torch
from torch import optim
from torch.utils.data import ConcatDataset

from utils import utils
from models.cl.continual_learner import CLBase


def train_cl(model, train_datasets, replay_mode="none", scenario="task", rnt=None, classes_per_task=None,
             iters=2000, batch_size=32, batch_size_replay=None, sample_in_model=False, only_last=False,
             generator=None, gen_iters=0, feedback=False, reinit=False, iters_first=None,
             args=None, loss_cbs=list(), eval_cbs=list(), gen_loss_cbs=list(), process_cbs=list()):
    freeze_convE = utils.checkattr(args, 'freeze_convE')

    device = args.device
    cuda = model._is_on_cuda()

    batch_size_replay = batch_size if batch_size_replay is None else batch_size_replay

    Generative = Current = Offline_TaskIL = False
    previous_model = None
    # generate the optimizer for model
    # optim_param_list = [
    #     {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
    # ]
    param_list = filter(lambda p: p.requires_grad, model.parameters())
    lr = args.lr
    if args.model_plat == "flymodel":
        model_op = None
        print("None optimiser is defined")
    elif args.optimiser == 'sgd':
        model_op = optim.SGD(param_list, lr=lr)
    elif args.optimiser == 'sgdm':
        model_op = optim.SGD(param_list, lr=lr, momentum=0.9)
    else:
        model_op = optim.Adam(param_list, lr=lr, betas=(0.9, 0.999))
    # Set optimizer(s) for generator

    if generator is not None:
        gen_optim_list = [
            {'params': filter(lambda p: p.requires_grad, generator.parameters()),
             'lr': args.lr_gen if hasattr(args, 'lr_gen') else args.lr},
        ]
        gen_optimizer = optim.Adam(gen_optim_list, betas=(0.9, 0.999))

    # Register starting param-values (needed for "intelligent synapses").
    if isinstance(model, CLBase) and model.si_c > 0:
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                model.register_buffer(f'{n}_SI_prev_task', p.detach().clone())

    for task_id, train_dataset in enumerate(train_datasets, 1):
        # task_id = 5
        # train_dataset = train_dataset[4]
        # If offline replay-setting, create large dataset of all tasks so far
        if replay_mode == 'offline' and (not scenario == 'task'):
            train_dataset = ConcatDataset(train_datasets[:task_id])
        # but when it is "offline" + 'task': all tasks so far should be visited seperatedly
        if replay_mode == 'offline' and scenario == 'task':
            Offline_TaskIL = True
            data_loader = [None] * task_id

        # prepare <dicts> to store running importance estimate the parameter-values before update
        if isinstance(model, CLBase) and model.si_c > 0:
            W = {}
            p_old = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    W[n] = p.data.clone().zero_()
                    p_old[n] = p.data.clone()

        # Find the classes participate in loss backward
        active_classes = None
        if scenario == 'task':
            active_classes = [list(range(classes_per_task*i, classes_per_task*(i+1))) for i in range(task_id)]
        elif scenario == 'class':
            active_classes = list(range(classes_per_task*task_id))

        # Reinitialize the model's parameters (if requested)
        if reinit:
            from utils.utils import init_params
            init_params(model, args)
            if generator is not None:
                init_params(generator, args)

        # DEFAULT SET THE GENERATOR'S BATCH SIZE AND ITERS ARE SAME AS MAIN MODEL
        iters_left = 1 if (not Offline_TaskIL) else [1]*task_id
        iters_main = iters_first if task_id == 1 else iters
        progress = tqdm.tqdm(range(1, iters_main+1))
        if generator is not None:
            progress_gen = tqdm.tqdm(range(1, iters_main + 1))

        for batch_index in range(1, iters_main+1):
            if not Offline_TaskIL:
                iters_left -= 1
                if iters_left == 0:
                    data_loader = iter(utils.get_data_loader(
                        train_dataset, batch_size, cuda=cuda,
                        drop_last=True, num_workers=args.num_workers
                    ))
                    iters_left = len(data_loader)
            else:
                # for the "offline replay" in Task-IL scenario, every single task learned before should be replayed
                batch_size_to_use = int(np.ceil(batch_size/task_id))
                for prev_id in range(task_id):
                    iters_left[prev_id] -= 1
                    if iters_left[prev_id] == 0:
                        data_loader[prev_id] = iter(utils.get_data_loader(
                            train_datasets[prev_id], batch_size_to_use, cuda=cuda,
                            drop_last=True, num_workers=args.num_workers
                        ))
                        iters_left[prev_id] = len(data_loader[prev_id])

            ################################################################
            # collect data for one batch training ############
            # CURRENT BATCH ##################
            if not Offline_TaskIL:
                x, y = next(data_loader)  # sample one data from data loader
                # Task-IL's labels should be range(classes_per_task) rather than plussing a constant
                y = y - classes_per_task * (task_id - 1) if scenario == 'task' else y
                x, y = x.to(device), y.to(device)
            else:
                x = y = task_used = None
                x_, y_ = list(), list()
                for prev_id in range(task_id):
                    x_temp, y_temp = next(data_loader[prev_id])
                    y_temp = y_temp - (classes_per_task * prev_id)
                    x_.append(x_temp.to(device))
                    if batch_size_to_use == 1:
                        y_temp = torch.Tensor([y_temp])
                    y_.append(y_temp.to(device))

            # the replay batch #################
            if not Offline_TaskIL and not Generative and not Current:
                x_ = y_ = scores_ = task_used = None  #-> when no replay

            # generate the input at first ########################################
            # Current Replay (Used for Lwf) ###################
            if Current:  # use the current task inputs as replay
                x_ = x
                task_used = None

            # task_id = 2
            # Generative = True
            # previous_generator = copy.deepcopy(generator).eval()
            # previous_model = copy.deepcopy(model).eval()
            # Generative Replay #################
            if Generative:
                # check whether to generate sample conditionally
                conditional_gen = True if (
                    (previous_generator.per_class and previous_generator.prior=="GMM") or
                    utils.checkattr(previous_generator, 'dg_gates')
                ) else False
                if conditional_gen and scenario == 'task':
                    # -if a conditional generator is used with task-IL scenario, generate data per previous task
                    x_ = list()
                    task_used = list()
                    single_task_batch = int(np.ceil(batch_size_replay / (task_id - 1)))
                    for prev_id in range(task_id - 1):
                        allowed_classes = list(range(prev_id*classes_per_task, (prev_id+1)*classes_per_task))
                        x_temp_ = previous_generator.sample(
                            single_task_batch, allowed_classes=allowed_classes, only_x=False
                        )  # -> [sample, y_used, task_used]
                        x_.append(x_temp_[0])
                        task_used.append(x_temp_[2])
                else:
                    allowed_classes = None if scenario == "domain" else list(range(classes_per_task*(task_id-1)))
                    allowed_domain = list(range(task_id-1))  # the allowed_domain is only relevant for "Domain" scenario
                    # x_ = model.generate_associate_samples()
                    x_temp_ = previous_generator.sample(
                        batch_size_replay, allowed_classes=allowed_classes, allowed_domain=allowed_domain,
                        only_x=False,
                    )  # -> [sample, y_used, task_used]
                    x_ = x_temp_[0]
                    task_used = x_temp_[2]

            # generate the output ########################################
            if Generative or Current:
                if scenario in ('domain', 'class') and previous_model.mask_dict is None:  # and previous_model.mask is None: which means not XdG
                    with torch.no_grad():
                        all_scores_ = previous_model.classify(x_, not_hidden=False if Generative else True)
                    scores_ = all_scores_[:, :(classes_per_task*(task_id-1))] if scenario == 'class' else all_scores_

                    _, y_ = torch.max(scores_, dim=1)
                else:
                    scores_ = list()
                    y_ = list()
                    if previous_model.mask_dict is None and not type(x_) == list:
                        with torch.no_grad():
                            all_scores_ = previous_model.classify(x_, not_hidden=False if Generative else True)
                    for prev_id in range(task_id - 1):
                        if previous_model.mask_dict is not None:
                            previous_model.apply_XdGmask(task=prev_id + 1)
                        if previous_model.mask_dict is not None or type(x_) == list:
                            with torch.no_grad():
                                if type(x_) == list:
                                    all_scores_ = previous_model.classify(
                                        x_[prev_id], not_hidden=False if Generative else True)
                                else:
                                    all_scores_ = previous_model.classify(
                                        x_, not_hidden=False if Generative else True)
                        temp_scores_ = all_scores_ if scenario == "domain" \
                            else all_scores_[:, (classes_per_task * prev_id):(classes_per_task * (prev_id + 1))]
                        scores_.append(temp_scores_)
                        _, temp_y_ = torch.max(temp_scores_, dim=1)
                        y_.append(temp_y_)

            # 这里还要检查采用的是软标签还是硬标签。
            if isinstance(model, CLBase):
                if model.replay_targets == 'soft':
                    y_ = None
                elif model.replay_targets == 'hard':
                    scores_ = None
                else:
                    raise ValueError(model.replay_target)

            ################################################################
            # train model(s) ###############################################
            # if batch_index <= iters_main
            loss_dict = model.train_a_batch(optimizer=model_op, x=x, y=y, x_=x_, y_=y_, scores_=scores_,
                                            tasks_=task_used, active_classes=active_classes, task=task_id,
                                            rnt=(1. if task_id == 1 else 1./task_id) if rnt is None else rnt,
                                            replay_not_hidden=False if Generative else True,
                                            freeze_convE=freeze_convE,
                                            one_task_ended=False if batch_index != iters_main else True)

            # under SI algorithm
            # Update running parameter importance estimates in W
            if isinstance(model, CLBase) and model.si_c > 0:
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            W[n].add_(-p.grad * (p.detach() - p_old[n]))
                        p_old[n] = p.detach().clone()

            ################################################################
            # activate the callbacks
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(progress, batch_index, loss_dict, task=task_id)
            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, batch_index, task=task_id)
            for process_cb in process_cbs:
                if process_cb is not None:
                    process_cb(model, batch_index, task=task_id)

            ################################################################
            # for generator-based CL here need to train the Generator
            if generator is not None:
                loss_dict_gen = generator.train_a_batch(
                    optimizer=gen_optimizer, x=x, y=y, x_=x_, y_=y_, scores_=scores_,
                    tasks_=task_used, active_classes=active_classes,
                    rnt=(1. if task_id == 1 else 1./task_id) if rnt is None else rnt,
                    task=task_id, freeze_convE=freeze_convE,
                    replay_not_hidden=False if Generative else True
                )

                for loss_cb in gen_loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress_gen, batch_index, loss_dict_gen, task=task_id)

        progress.close()
        if generator is not None:
            progress_gen.close()

        # update after finishing each task
        # EWC: estimate Fisher Information matrix and update term for quadratic penalty
        if isinstance(model, CLBase) and model.ewc_lambda > 0:
            allowed_classes = list(
                range(classes_per_task * (task_id-1), classes_per_task * task_id)
            ) if scenario == 'task' else (list(range(classes_per_task*task_id)) if scenario=="class" else None)
            if model.mask_dict is not None:
                model.apply_XdGmask(task=task_id)  # only use to estimate FI-matrix
            model.estimate_fisher(train_dataset, allowed_classes=allowed_classes)

        if isinstance(model, CLBase) and model.mas_lambda > 0:
            allowed_classes = list(
                range(classes_per_task * (task_id-1), classes_per_task * task_id)
            ) if scenario == 'task' else (list(range(classes_per_task*task_id)) if scenario=='class' else None)
            if model.mask_dict is not None:
                model.apply_XdGmask(task=task_id)
            model.estimate_mas_importance(train_dataset, allowed_classes=allowed_classes)

        if isinstance(model, CLBase) and model.si_c > 0:
            model.update_omega(W, model.epsilon)

        if replay_mode == 'generative':
            previous_generator = previous_model if feedback else copy.deepcopy(generator).eval()
            previous_model = copy.deepcopy(model).eval()
            Generative = True
        elif replay_mode == 'current':
            previous_model = copy.deepcopy(model).eval()
            Current = True


def train(model, train_loader, iters, save_every=None, m_dir='/models', args=None, model_name=None,
          loss_cbs=list(), eval_cbs=list()):
    device = args.device
    # freeze_convE = (utils.checkattr(args, "freeze_convE") and hasattr(args, "depth") and args.depth > 0)

    optim_param_list = [
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr':args.lr},
    ]
    model_op = optim.Adam(optim_param_list, betas=(0.9, 0.999))

    bar = tqdm.tqdm(total=iters)
    iteration = epoch = 0
    while iteration < iters:
        epoch += 1

        for batch_idx, (data, y) in enumerate(train_loader):
            iteration += 1

            data, y = data.to(device), y.to(device)
            loss_dict = model.train_a_batch(optimizer=model_op, x=data, y=y)

            # evaluation call back
            for loss_cb in loss_cbs:
                if loss_cb is not None:
                    loss_cb(bar, iteration, loss_dict, epoch=epoch)

            for eval_cb in eval_cbs:
                if eval_cb is not None:
                    eval_cb(model, iteration, epoch=epoch)

            if iteration == iters:
                bar.close()
                break

            if (save_every is not None) and (iteration % save_every) == 0:
                utils.save_checkpoint(model, model_dir=m_dir, args=args, name=model_name)


