from torch.utils.data.dataloader import default_collate
from utils.utils import checkattr
from utils.utils import get_neuron_tag, get_surr_tag

from data_utils.utils import get_multitask_experiment
from spikingjelly.activation_based import neuron


################################################################
# genarate the parameters stamp ###############################
def get_param_stamp_from_args(args):
    from utils.model_definition import define_specific_model
    config = get_multitask_experiment(
        name=args.experiment, scenario=args.scenario, tasks=args.tasks, data_dir=args.data_dir, only_config=True,
        normalize=not args.no_norm, verbose=False, T=args.T
    )
    model = define_specific_model(args=args, config=config, device='cpu')
    if checkattr(args, 'feedback'):
        model.lamda_pl = 1. if not hasattr(args, 'pl') else args.pl

    model_name = model.name
    param_stamp = get_param_stamp(args, model_name, replay=(hasattr(args, 'replay') and not args.replay == 'none'),
                                  verbose=False)
    return param_stamp


# generate a file name according to the parameters
def get_param_stamp(args, model_name, verbose=True, replay=False):
    multi_n_stamp = f"-{args.tasks}-{args.scenario}" if hasattr(args, 'tasks') else ""
    task_stamp = f"{args.experiment}" \
                 f"{'-N'if not args.no_norm else''}" \
                 f"{'+' if hasattr(args, 'augment') and args.augment else ''}{multi_n_stamp}"
    if verbose:
        print('--> task:  ' + task_stamp)

    model_stamp = model_name
    if verbose:
        print('--> model:  '+model_stamp)

    # -for hyper-parameters
    pre_conv = ""
    if (checkattr(args, "pre_convE") or checkattr(args, "pre_convD")) and (hasattr(args, 'depth') and args.depth > 0):
        ltag = "" if not hasattr(args, "convE_ltag") or args.convE_ltag=='none' else args.convE_ltag
        pre_conv = f"-pCvE{ltag}" if args.pre_convE else "-pCvD"
    freeze_conv = ""
    if (checkattr(args, "freeze_convD") or checkattr(args, "freeze_convE")) and hasattr(args, 'depth') and args.depth>0:
        freeze_conv = "-fCvE" if checkattr(args, "freeze_convE") else "-fCvD"
        freeze_conv = "-fConv" if checkattr(args, "freeze_convE") and checkattr(args, "freeze_convD") else freeze_conv
    neuron_stamp = "-{neuron_tag}-surr_{surr_tag}-t{T}".format(
        neuron_tag=get_neuron_tag(args.fc_neu), surr_tag=get_surr_tag(args.surrogate), T=args.T
    )
    loss_stamp = f"-loss_tet-lslb_{args.loss_lambda}-lsmean_{args.loss_means}" if args.loss_type == 'tet' else "-loss_ce"
    hyper_stamp = "{i_e}{num}-lr{lr}{lrg}-b{bsz}{pretr}{freeze}{reinit}{neuron}{optim}{loss}".format(
        i_e="e" if args.iters is None else "i",
        num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        lrg=("" if args.lr == args.lr_gen else "-lrG{}".format(args.lr_gen)) if (
            hasattr(args, "lr_gen") and hasattr(args, "replay") and args.replay=="generative" and
            (not checkattr(args, "feedback"))
        ) else "",
        bsz=args.batch, pretr=pre_conv, freeze=freeze_conv, reinit="-R" if checkattr(args, 'reinit') else "",
        neuron=neuron_stamp, optim=f'-{args.optimiser}', loss=loss_stamp
    )
    if verbose:
        print("--> hyper-params:  " + hyper_stamp)

    # -for EWC / SI
    if (checkattr(args, 'ewc') and args.ewc_lambda>0) or (checkattr(args, 'si') and args.si_c>0) or (args.mas and args.mas_lambda > 0):
        ewc_stamp = "EWC{l}{simpled_tag}-{fi}{o}".format(
            l=args.ewc_lambda, fi="{}".format("N" if args.fisher_n is None else args.fisher_n),
            simpled_tag=f"sim{args.ewc_beta}" if args.simpled_EWC else "",
            o="-O{}".format(args.gamma) if checkattr(args, 'online') else "",
        ) if (checkattr(args, 'ewc') and args.ewc_lambda>0) else ""
        si_stamp = "SI{c}-{eps}".format(c=args.si_c, eps=args.epsilon) if (checkattr(args,'si') and args.si_c>0) else ""
        mas_stamp = f"MAS{args.mas_lambda}{'-o' if args.online_mas else ''}" if args.mas == True and args.mas_lambda > 0 else ""
        both = "--" if (checkattr(args,'ewc') and args.ewc_lambda>0) and (checkattr(args,'si') and args.si_c>0) else ""
        if verbose and checkattr(args, 'ewc') and args.ewc_lambda>0:
            print(" --> EWC:           " + ewc_stamp)
        if verbose and checkattr(args, 'si') and args.si_c>0:
            print(" --> SI:            " + si_stamp)
    ewc_stamp = "--{}{}{}{}".format(ewc_stamp, both, si_stamp, mas_stamp) if (
            (checkattr(args, 'ewc') and args.ewc_lambda>0) or (checkattr(args, 'si') and args.si_c>0) or (args.mas and args.mas_lambda > 0)
    ) else ""

    # -for XdG
    xdg_stamp = ""
    if (checkattr(args, "xdg") and args.xdg_prop > 0):
        xdg_stamp = "--XdG{}".format(args.xdg_prop)
        if verbose:
            print(" --> XdG:           " + "gating = {}".format(args.xdg_prop))

    if replay:
        replay_stamp = "{H}{rep}{bat}{distil}".format(
            H="" if not args.replay == "generative" else "H",
            rep="gen" if args.replay == "generative" else args.replay,
            bat="" if (
                    (not hasattr(args, 'batch_replay')) or (
                        args.batch_replay is None) or args.batch_replay == args.batch
            ) else "-br{}".format(args.batch_replay),

            distil="-Di{}".format(args.temp) if args.distill else "",
        )
        if verbose:
            print(" --> replay:        " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""
    # -for choices regarding reconstruction loss

    if checkattr(args, "feedback"):
        recon_stamp = "--{}{}".format(
            "H_" if checkattr(args, "hidden") and hasattr(args, 'depth') and args.depth>0 else "", args.recon_loss
        )
    elif hasattr(args, "replay") and args.replay=="generative":
        recon_stamp = "--{}".format(args.recon_loss)
    else:
        recon_stamp = ""

    param_stamp = "{}--{}--{}{}{}{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, ewc_stamp, xdg_stamp, replay_stamp,
        recon_stamp, "-s{}".format(args.seed) if not args.seed==0 else"",
    )
    if verbose:
        print(param_stamp)

    return param_stamp
