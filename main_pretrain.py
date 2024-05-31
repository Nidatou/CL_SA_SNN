import os
import numpy as np
import torch

from data_utils.utils import get_singletask_experiment
from utils import utils, options, model_definition, param_stamps
from evaluations import evaluate, callbacks
from visual import plotting
import train

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

def run(args):

    cuda = torch.cuda.is_available()
    # device = torch.device(args.device if cuda else "cpu")
    device = torch.device(args.device if cuda else "cpu")
    args.device = device
    # print(f"The model deployed on {device}")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # Create plots-directory if needed
    if args.pdf and not os.path.isdir(args.plot_dir):
        os.mkdir(args.plot_dir)

    #########################################################
    # Data preparation #################################
    print("\nPreparing the data...")
    (train_set, test_set), config = get_singletask_experiment(
        name=args.experiment, data_dir=args.data_dir, verbose=True,
        normalize=not args.no_norm,
        augment=True if utils.checkattr(args, "augment") else False,
    )

    # Specify "data-loader" (among others for easy random shuffling and 'batchifying')
    train_loader = utils.get_data_loader(train_set, batch_size=args.batch, cuda=cuda, drop_last=True)

    # Determine number of iterations / epochs:
    iters = args.iters if args.iters else args.epochs * len(train_loader)
    epochs = ((args.iters - 1) // len(train_loader)) + 1 if args.iters else args.epochs

    #########################################################
    # Main model ######################################

    # Specify model
    print("\nDefining the model...")
    pre_model = model_definition.define_classifier(args=args, config=config, device=device)
    # pre_model = torch.nn.DataParallel(pre_model)
    # pre_model = pre_model.to(device)

    # Initialize (pre-trained) parameters
    pre_model = utils.init_params(pre_model, args)
    if utils.checkattr(args, "freeze_full"):
        for param in pre_model.parameters():
            param.requires_grad = False
        for param in pre_model.classifier.parameters():
            param.requires_grad = True

    # Set optimizer
    optim_list = [{'params': filter(lambda p: p.requires_grad, pre_model.parameters()), 'lr': args.lr}]
    pre_model.optimizer = torch.optim.Adam(optim_list, betas=(0.9, 0.999))

    #########################################################
    # Evaluation setting ####################################

    # Get parameter-stamp
    print("\nParameter-stamp...")
    param_stamp = param_stamps.get_param_stamp(args, pre_model.name, verbose=True)

    # Define [progress_dicts] to keep track of performance during training for storing and for later plotting in pdf
    progress_dict = evaluate.initiate_progress_dict(n_tasks=1)

    # Prepare for plotting in visdom
    visdom = None
    # --------------------------------------

    #########################################################
    # Callbacks #####################################
    # Determine after how many iterations to evaluate the model
    eval_log = args.acc_log if (args.acc_log is not None) else len(train_loader)

    # Define callback-functions to evaluate during training
    # -loss
    loss_cbs = [callbacks._solver_loss_cb(log_interval=args.loss_log, visdom=visdom, epochs=epochs)]
    # -accuracy
    eval_cb = callbacks._eval_cb(log_interval=eval_log, test_datasets=[test_set], visdom=visdom,
                                 progress_dict=progress_dict, verbose=True,)
    # -visualize extracted representation
    # -----------------------------------------------------------

    #########################################################
    # Training ######################################

    # (Pre)train model
    print("\nTraining...")
    train.train(pre_model, train_loader, iters, loss_cbs=loss_cbs, eval_cbs=[eval_cb],
                save_every=1000 if args.save else None, m_dir=args.model_dir, args=args)
    # Save (pre)trained model
    if args.save:
        # -conv part
        save_name = pre_model.convE.name if (
            not hasattr(args, 'convE_stag') or args.convE_stag == "none"
        ) else "{}-{}".format(pre_model.convE.name, args.convE_stag)
        utils.save_checkpoint(pre_model.convE, args.model_dir, name=save_name)
        # -full part
        save_name = pre_model.name if (
                not hasattr(args, 'full_stag') or args.full_stag == "none"
        ) else "{}-{}".format(pre_model.name, args.full_stag)
        utils.save_checkpoint(pre_model, args.model_dir, name=save_name)

    # if requested, generate pdf.
    if args.pdf:
        # -open pdf
        plot_name = "{}/{}.pdf".format(args.plot_dir, param_stamp)
        pp = plotting.open_pdf(plot_name)
        # -Fig1: show some images
        images, _ = next(iter(train_loader))  # --> get a mini-batch of random training images
        plotting.plot_images_from_tensor(images, pp, title="example input images", config=config)
        # -Fig2: accuracy
        figure = plotting.plot_lines(
            progress_dict["all_tasks"], x_axes=progress_dict["x_iteration"],
            line_names=['ave accuracy'], xlabel="Iterations", ylabel="Test accuracy")
        pp.savefig(figure)
        # -close pdf
        pp.close()
        # -print name of generated plot on screen
        print("\nGenerated plot: {}\n".format(plot_name))


if __name__ == "__main__":
    args = options.handle_inputs(
        single_task=True,
        not_include=['perm', 'alloc'], filename='main_pretrain',
        description='Train classifier for pretraining conv-layers.'
    )
    args.loss_type = 'tet'
    args.experiment = "CIFAR100"
    args.epochs = 200
    args.lr = 1e-3
    args.convE_stag = "res_C100"
    args.augment = True
    args.no_norm = False
    args.conv_type = 'resNet'
    args.surrogate = 'ATAN'
    args.optimiser = 'adam'
    args.n_blocks = [3, 3, 2]
    args.depth = 4
    args.fc_depth = 2
    args.fc_units = 256
    args.batch = 128
    args.channels = 64
    args.seed = 3434
    args.loss_lambda = 5e-2
    args.acc_log = 1000
    args.loss_log = 1000
    args.gp = True
    args.conv_neu = 'LIF'
    args.fc_neu = 'LIF'
    # args.experiment = "MNIST"
    args = options.preprocess_args(args, single_task=True)
    run(args)

