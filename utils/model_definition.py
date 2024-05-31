from models import utils
from utils.utils import checkattr


# -------------------------------------------------------------------------------------------------------------------##
# Function for defining classifier model
def define_classifier(args, config, device):
    # -import required model
    from models.snn_classifier import SNN_Classifier
    # -create model
    if hasattr(args, "depth") and args.depth>0:
        model = SNN_Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'], T=args.T,
            # -conv-layers
            conv_type=args.conv_type, depth=args.depth, start_channels=args.channels, reducing_layers=args.rl,
            num_blocks=args.n_blocks, conv_bn=args.conv_bn, conv_neu=args.conv_neu, global_pooling=checkattr(args, 'gp'),
            no_neu=False,
            # -fc-layers
            fc_depth=args.fc_depth, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_neu=args.fc_neu, gate_buffer=True,
            # -training-specific components
            hidden=checkattr(args, 'hidden'), surr_func=args.surrogate,
            loss_type=args.loss_type, loss_lambda=args.loss_lambda, loss_means=args.loss_means,
        ).to(device)
    else:
        model = SNN_Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'], T=args.T,
            # -fc-layers
            fc_depth=args.fc_depth, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_neu=args.fc_neu, gate_buffer=True,
            # -training-specific components
            hidden=checkattr(args, 'hidden'), surr_func=args.surrogate,
            loss_type=args.loss_type, loss_lambda=args.loss_lambda, loss_means=args.loss_means,
        ).to(device)
    # -return model
    return model


def define_SDM_classifier(args, config, device):
    from models.SDMLP import SDM
    model = SDM(
        input_size=config['channels'] * config['size']**2, nneu=args.h_dim, output_size=config['classes'], device=device,
        # -> top-k hyper parameters
        k_mask=args.use_mask, k_approach=args.k_approach, k_min=args.k_min, k_max=args.k_max,
        k_trans_ep=args.k_trans_ep, gaba_switch_num=args.gaba_switch_num,
        # -> parameter_regularisation
        norm_ad=args.norm_ad, norm_val=args.norm_val, dale=args.dale
    ).to(device)
    return model


def define_Snn_WTAk(args, config, device):
    # from models.snn_kWTA import Gate_snn
    from models.snn_Adaptive_kWTA import Gate_snn
    from models.SDMLP import SDM
    if args.use_ann:
        model = SDM(
            input_size=config['channels'] * config['size'] ** 2, nneu=args.h_dim, output_size=config['classes'],
            device=device,
            k_mask=args.use_mask, k_approach=args.k_approach, k_min=args.k_min, k_max=None,
            k_trans_ep=args.k_trans_ep, gaba_switch_num=args.gaba_switch_num, use_topK=args.topk,
            # -> parameter_regularisation
            norm_ad=args.norm_ad, norm_val=args.norm_val, dale=args.dale,
        ).to(device)
    else:
        if not args.topk:
            model = define_classifier(args, config, device)
        else:
            if args.depth is None:
                args.depth = 0
            model = Gate_snn(
                image_size=config['size'], image_channels=config['channels'], output_size=config['classes'], T=args.T,
                device=device, surr_func=args.surrogate,
                ########################################################
                # conv-parts
                conv_type=args.conv_type, depth=args.depth, start_channels=args.channels, reducing_layers=args.rl,
                num_blocks=args.n_blocks, conv_bn=args.conv_bn, conv_neu=args.conv_neu, global_pooling=checkattr(args, 'gp'),
                fnl=True,
                ########################################################
                # fc_kWTA_layers-parts
                h_dim=args.h_dim, fc_neu=args.neu,
                k_min=args.k_min, use_mask=args.use_mask, step_mask=args.step_mask, loosen_tr=args.loosen_tr,
                # -> parameters for trace mask
                tr_decay=args.tr_decay, hard_mask=args.hard_mask,
                # -> parameters for adaptive threshold
                adaptive_thresh=args.adaptive_thresh, adap_param=args.adap_param, adap_approach=args.adap_approach,
                thresh_max=args.thresh_max, thresh_min=args.thresh_min,
                # -> parameters for k selection
                last_inhibit=args.last_inhibit, curr_excite=args.curr_excite, tune_approach=args.tune_approach,
                inhibit_p=args.inhibit_p, excite_p=args.excite_p, tune_param=args.tune_param, adapt_tune=args.adap_curr,
                k_param=args.k_param, k_approach=args.k_approach, k_max=args.k_max,
                # ->other parameters
                dale=args.dale, norm_input=args.norm_input, norm_fir=args.norm_ad, final_neu=args.fnl,
                loss_type=args.loss_type, loss_lamb=args.loss_lambda, loss_means=args.loss_means,
                spk_ipt=args.spk_ipt, filter_prop=args.filter_prop, norm_const=args.norm_const,
            ).to(device)
    return model


def define_sleep_model(args, config, device):
    from models.Sleep_model import Sleep_snn
    model = Sleep_snn(
        input_size=config['channels'] * config['size'] ** 2, output_size=config['classes'], T=args.T, device=device,
        surr_func=args.surrogate, neu=args.neu,
        k_min=args.k_min, step_mask=args.step_mask, tr_decay=args.tr_decay, hard_mask=args.hard_mask,
        # custom parameter for wta_k
        adaptive_thresh=args.adaptive_thresh, adap_param=args.adap_param, adap_approach=args.adap_approach,
        thresh_max=args.thresh_max, thresh_min=args.thresh_min,
        # -> parameters for k selection
        last_inhibit=args.last_inhibit, curr_excite=args.curr_excite, inhibit_p=args.inhibit_p, excite_p=args.excite_p,
        tune_param=args.tune_param, adapt_tune=args.adap_curr,
        # fc_part
        h_dim=args.h_dim, final_neu=args.fnl,
        # sleep unsupervised learning
        inc_lr=args.inc_lr, dec_lr=args.dec_lr, sleep_times=args.sleep_times, sleep_batch=args.sleep_batch,
        need_sleep=args.need_sleep,
        # other setting
        loss_type=args.loss_type, loss_lamb=args.loss_lambda, loss_means=args.loss_means,
        dale=args.dale, norm_weight=args.norm_ad, norm_input=args.norm_input, norm_const=args.norm_const,
        filter_prop=args.filter_prop, spk_ipt=args.spk_ipt, topk_sleep=args.topk_sleep
    ).to(device)
    return model


def define_test_model(args, config, device):
    from models.solid_SDM import Solid_Gate
    model = Solid_Gate(
        image_size=config['size'], image_channels=config['channels'], output_size=config['classes'], T=args.T,
        device=args.device, surr_func=args.surrogate,
        ######################################################
        # fc_kWTA_layers-parts
        h_dim=args.h_dim, fc_neu=args.neu, k_min=args.k_min, step_mask=args.step_mask,
        # -> parameters for trace mask
        tr_decay=args.tr_decay, hard_mask=args.hard_mask,
        # -> parameters for adaptive threshold
        adaptive_thresh=args.adaptive_thresh, adap_param=args.adap_param, adap_approach=args.adap_approach,
        thresh_max=args.thresh_max, thresh_min=args.thresh_min,
        # -> parameters for k selection
        # -> other parameters
        dale=args.dale, norm_input=args.norm_input, norm_fir=args.norm_ad, final_neu=args.fnl,
        loss_type=args.loss_type, loss_lamb=args.loss_lambda, loss_means=args.loss_means, norm_const=args.norm_const
    ).to(device)
    return model


def define_specific_model(args, config, device):
    if args.model_plat == 'standard':
        model = define_Snn_WTAk(args, config, device)
    elif args.model_plat == 'flymodel':
        from models.FlyModel_whole import FlyModel_Whole
        from models.FlyModel import FlyModel
        model = FlyModel_Whole(
            image_size=config['size'], image_channels=config['channels'], output_size=config['classes'], device=device,
            # the parameters for fly-model construction
            n_kc=args.n_kc, kc_response=args.kc_response, k_num=args.k_num, fly_lr=args.lr, min_max=args.min_max
        ).to(device)
    elif args.model_plat == 'sleep':
        model = define_sleep_model(args, config, device)
    elif args.model_plat == 'test':
        model = define_test_model(args, config, device)
    else:
        raise NotImplementedError(f'<{args.model_plat}> is not defined in the projects')

    return model
