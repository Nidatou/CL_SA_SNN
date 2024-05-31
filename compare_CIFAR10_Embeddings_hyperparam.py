#!/usr/bin/env python3
import os
import numpy as np
from utils.param_stamps import get_param_stamp_from_args
from utils import options, utils
from visual import plotting
from matplotlib.pyplot import get_cmap
import main_exp


## Parameter-values to compare
# lamda_list = [100, 200, 400]
# lamda_list = [1000, 4000, 10000]
lambda_list = [200, 400, 2000, 4000, 10000]
beta_list = [0.08]
# gaba_list = [3000000, 5000000]
# gaba_list = [2000000, 4000000, 6000000, 8000000]
mas_lambda_list = [0.5, 1, 4, 10, 100, 200, 400, 1000, 10000]

k_reponse_list = [32, 64]


def get_result(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run, and if not do so
    if os.path.isfile('{}/acc-{}.txt'.format(args.results_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main_exp.run(args)
    # -get average accuracies
    fileName = '{}/acc-{}.txt'.format(args.results_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -return it
    return ave


if __name__ == '__main__':

    # Load input-arguments & set default values
    args = options.handle_inputs(
        single_task=False, not_include=["perm"], filename="_compare_CIFAR10",
        description="Compare performance of continual learning strategies on different scenarios of split CIFAR-10."
    )

    ################################################################################
    # Add default arguments (will be different for different runs)
    args.ewc = False
    args.online = False
    args.si = False
    args.xdg = False
    args.iters = 20000
    args.nneu = 1000
    args.lr = 5e-2
    args.optimiser = 'sgd'
    args.scenario = 'class'
    # args.experiment = "splitMNIST"
    args.experiment = "CIFAR10_Embeddings_0"

    # Use pre-trained convolutional layers for all compared methods
    args.topk = True
    # args.pre_convE = True
    args.pre_convE = False
    args.freeze_convE = True
    args.use_mask = True
    args.hard_mask = False
    args.use_ann = False
    args.augment = True
    args.no_norm = False
    args.model_plat = 'standard'
    args.surrogate = "ATAN"
    args.record_process = False

    ## Select list for SI based on chosen scenario
    options.preprocess_args(args, single_task=False)

    ## If needed, create plotting directory
    if not os.path.isdir(args.plot_dir):
        os.mkdir(args.plot_dir)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#
    args.neu = 'if'
    args.adaptive_thresh = True
    args.adap_param = 2000000
    args.fnl = "normal"
    args.k_min = 10
    # ###########################################################
    # # SNN mask ################################################
    # ###########################################################
    # args.step_mask = False
    # # Baseline
    # M_BASE = get_result(args)
    #
    # # Simpled EWC
    # M_EWC = {}
    # args.ewc = True
    # args.simpled_EWC = True
    # for ewc_beta in beta_list:
    #     M_EWC[ewc_beta] = {}
    #     args.ewc_beta = ewc_beta
    #     for ewc_lambda in lamda_list:
    #         args.ewc_lambda = ewc_lambda
    #         M_EWC[ewc_beta][ewc_lambda] = get_result(args)
    # args.ewc = False
    # args.simpled_EWC = False

    # ###########################################################
    # # SNN mask poi ############################################
    # ###########################################################
    # args.step_mask = False
    # args.spk_ipt = True
    # args.filter_prop = 0.1
    # args.norm_const = 10.
    # # Baseline
    # MP_BASE = get_result(args)
    #
    # # Simpled EWC
    # MP_EWC = {}
    # args.ewc = True
    # args.simpled_EWC = True
    # for ewc_beta in beta_list:
    #     MP_EWC[ewc_beta] = {}
    #     args.ewc_beta = ewc_beta
    #     for ewc_lambda in lamda_list:
    #         args.ewc_lambda = ewc_lambda
    #         MP_EWC[ewc_beta][ewc_lambda] = get_result(args)
    # args.ewc = False
    # args.simpled_EWC = False
    # args.spk_ipt = False
    # args.filter_prop = 1.
    # args.norm_const = 1.

    # ###########################################################
    # # SNN step mask ###########################################
    # ###########################################################
    # args.step_mask = True
    # args.iters = 20000
    # args.adap_param = 2000000
    # # Baseline
    # MS_BASE = get_result(args)
    #
    # # Simpled EWC
    # MS_EWC = {}
    # args.ewc = True
    # args.simpled_EWC = True
    # for ewc_beta in beta_list:
    #     MS_EWC[ewc_beta] = {}
    #     args.ewc_beta = ewc_beta
    #     for ewc_lambda in lamda_list:
    #         args.ewc_lambda = ewc_lambda
    #         MS_EWC[ewc_beta][ewc_lambda] = get_result(args)
    # args.ewc = False
    # args.simpled_EWC = False
    # args.iters = 20000
    # args.adap_param = 2000000

    # ###########################################################
    # # FlyModel ################################################
    # ###########################################################
    # args.model_plat = 'flymodel'
    # args.iters = 40
    # args.n_kc = 1000
    # args.k_num = 10
    # args.lr = 0.005
    # args.min_max = True
    # FLY_1K = {}
    # # get_result(args)
    # for kc_response in k_reponse_list:
    #     args.kc_response = kc_response
    #     FLY_1K[kc_response] = get_result(args)
    #
    # args.n_kc = 10000
    # args.k_num = 100
    # FLY_10K = {}
    # for kc_response in k_reponse_list:
    #     args.kc_response = kc_response
    #     FLY_10K[kc_response] = get_result(args)
    #
    # args.model_plat = 'standard'
    # args.iters = 20000

    # ###########################################################
    # # ANN_SDM #################################################
    # ###########################################################
    # args.model_plat = 'standard'
    # args.k_approach = "GABA_SWITCH"
    # args.use_mask = False
    # args.topk = True
    # args.use_ann = True
    # args.seed = 12
    # SDM = {}
    # for gaba_val in gaba_list:
    #     args.gaba_switch_num = gaba_val
    #     SDM[gaba_val] = get_result(args)
    #
    # args.use_mask = True
    # args.use_ann = False
    # args.iters = 20000

    # ###########################################################
    # # ANN_SDM + EWC ###########################################
    # ###########################################################
    # args.model_plat = 'standard'
    # args.k_approach = "GABA_SWITCH"
    # args.use_mask = False
    # args.topk = True
    # args.use_ann = True
    # args.seed = 11
    # args.ewc = True
    # args.simpled_EWC = True
    # args.ewc_lambda = 100
    # args.ewc_beta = 0.08
    # SDM_EWC = {}
    # for gaba_val in gaba_list:
    #     args.gaba_switch_num = gaba_val
    #     SDM_EWC[gaba_val] = get_result(args)
    #
    # args.use_mask = True
    # args.use_ann = False
    # args.k_approach = "FLAT"

    # ###########################################################
    # # ANN_SDM + EWC2 ##########################################
    # ###########################################################
    # args.experiment = "CIFAR10_Embeddings_0"
    # args.k_approach = "GABA_SWITCH"
    # args.use_mask = False
    # args.topk = True
    # args.use_ann = True
    # args.seed = 11
    # args.ewc = True
    # args.simpled_EWC = True
    # args.ewc_lambda = 100
    # args.ewc_beta = 0.08
    # SDM_EWC_O = {}
    # for gaba_val in gaba_list:
    #     args.gaba_switch_num = gaba_val
    #     SDM_EWC_O[gaba_val] = get_result(args)

    args.use_mask = True
    args.use_ann = False
    args.experiment = "CIFAR10_Embeddings_0"
    args.k_approach = "FLAT"

    ###########################################################
    # Simpled_EWC #############################################
    ###########################################################
    args.topk = False
    S_EWC = {}
    args.ewc = True
    args.simpled_EWC = True
    args.fc_depth = 2
    args.fc_units = 1000
    args.h_dim = 1000
    for ewc_beta in beta_list:
        S_EWC[ewc_beta] = {}
        args.ewc_beta = ewc_beta
        for ewc_lambda in lambda_list:
            args.ewc_lambda = ewc_lambda
            print(f"now EWC_BETA={ewc_beta}, EWC_LAMBDA={ewc_lambda}")
            S_EWC[ewc_beta][ewc_lambda] = get_result(args)

    args.topk = True
    args.ewc = False
    args.simpled_EWC = False

    ###########################################################
    # MAS #####################################################
    ###########################################################
    args.topk = False
    MAS = {}
    args.mas = True
    args.iters = 100
    args.fc_depth = 2
    args.fc_units = 1000
    args.h_dim = 1000
    for mas_lambda in mas_lambda_list:
        args.mas_lambda = mas_lambda
        MAS[mas_lambda] = get_result(args)

    args.topk = True
    args.mas = False

    # #-------------------------------------------------------------------------------------------------#
    #
    # #--------------------------------------------#
    # #----- COLLECT DATA AND PRINT ON SCREEN -----#
    # #--------------------------------------------#
    #
    # ext_c_list = [0] + c_list
    # ext_lambda_list = [0] + lamda_list
    # ext_xdg_list = [0] + xdg_list
    # print("\n")
    #
    #
    # ###---EWC + online EWC---###
    #
    # # -collect data
    # ave_acc_ewc = [BASE] + [EWC[ewc_lambda] for ewc_lambda in lamda_list]
    # ave_acc_per_lambda = [ave_acc_ewc]
    # for gamma in gamma_list:
    #     ave_acc_temp = [BASE] + [OEWC[gamma][ewc_lambda] for ewc_lambda in lamda_list]
    #     ave_acc_per_lambda.append(ave_acc_temp)
    # # -print on screen
    # print("\n\nELASTIC WEIGHT CONSOLIDATION (EWC)")
    # print(" param-list (lambda): {}".format(ext_lambda_list))
    # print("  {}".format(ave_acc_ewc))
    # print("--->  lambda = {}     --    {}".format(ext_lambda_list[np.argmax(ave_acc_ewc)], np.max(ave_acc_ewc)))
    # if len(gamma_list) > 0:
    #     print("\n\nONLINE EWC")
    #     print(" param-list (lambda): {}".format(ext_lambda_list))
    #     curr_max = 0
    #     for gamma in gamma_list:
    #         ave_acc_temp = [BASE] + [OEWC[gamma][ewc_lambda] for ewc_lambda in lamda_list]
    #         print("  (gamma={}):   {}".format(gamma, ave_acc_temp))
    #         if np.max(ave_acc_temp) > curr_max:
    #             gamam_max = gamma
    #             lamda_max = ext_lambda_list[np.argmax(ave_acc_temp)]
    #             curr_max = np.max(ave_acc_temp)
    #     print("--->  gamma = {}  -  lambda = {}     --    {}".format(gamam_max, lamda_max, curr_max))
    #
    #
    # ###---SI---###
    #
    # # -collect data
    # ave_acc_si = [BASE] + [SI[c] for c in c_list]
    # # -print on screen
    # print("\n\nSYNAPTIC INTELLIGENCE (SI)")
    # print(" param list (si_c): {}".format(ext_c_list))
    # print("  {}".format(ave_acc_si))
    # print("---> si_c = {}     --    {}".format(ext_c_list[np.argmax(ave_acc_si)], np.max(ave_acc_si)))
    #
    #
    # ###---XdG---###
    #
    # if args.scenario == "task":
    #     # -collect data
    #     ave_acc_xdg = [BASE] + [XDG[c] for c in xdg_list]
    #     # -print on screen
    #     print("\n\nCONTEXT-DEPENDENT GATING (XDG))")
    #     print(" param list (gating_prop): {}".format(ext_xdg_list))
    #     print("  {}".format(ave_acc_xdg))
    #     print("---> gating_prop = {}     --    {}".format(ext_xdg_list[np.argmax(ave_acc_xdg)], np.max(ave_acc_xdg)))
    # print('\n')
    #
    #
    # ###---Brain-Inspired Replay (BI-R)---###
    #
    # # -collect data
    # ave_acc_bir = [BIR[dg_prop] for dg_prop in dg_prop_list_onlybir]
    # # -print on screen
    # print("\n\nBRAIN-INSPIRED REPLAY (BI-R)")
    # print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
    # print("  {}".format(ave_acc_bir))
    # print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir)], np.max(ave_acc_bir)))
    #
    #
    # ###---BI-R + SI---###
    #
    # # -collect data
    # ave_acc_bir_per_c = []
    # for dg_prop in dg_prop_list:
    #     ave_acc_bir_per_c.append([BIR_SI[dg_prop][c] for c in ext_c_list])
    # # -print on screen
    # print("\n\nBI-R & SI")
    # print(" param-list (si_c): {}".format(ext_c_list))
    # curr_max = 0
    # for dg_prop in dg_prop_list:
    #     ave_acc_temp = [BIR_SI[dg_prop][c] for c in ext_c_list]
    #     print("  (dg-prop={}):   {}".format(dg_prop, ave_acc_temp))
    #     if np.max(ave_acc_temp)>curr_max:
    #         dg_prop_max = dg_prop
    #         si_max = ext_c_list[np.argmax(ave_acc_temp)]
    #         curr_max = np.max(ave_acc_temp)
    # print("--->  dg_prop = {}  -  si_c = {}     --    {}".format(dg_prop_max, si_max, curr_max))
    #
    #
    # if args.per_bir_comp:
    #
    #     ###---BI-R per component---###
    #
    #     # -collect data
    #     ave_acc_bir_no_rtf = [BIR_no_RTF[dg_prop] for dg_prop in dg_prop_list_onlybir]
    #     ave_acc_bir_no_con = [BIR_no_CON[dg_prop] for dg_prop in dg_prop_list_onlybir]
    #     ave_acc_bir_no_int = [BIR_no_INT[dg_prop] for dg_prop in dg_prop_list_onlybir]
    #     ave_acc_bir_no_dis = [BIR_no_DIS[dg_prop] for dg_prop in dg_prop_list_onlybir]
    #     ave_acc_gr_plus_gat = [GR_plus_GAT[dg_prop] for dg_prop in dg_prop_list_onlybir]
    #     # -print on screen
    #     print("\n\nBI-R minus REPLAY-THROUGH-FEEDBACK")
    #     print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
    #     print("  {}".format(ave_acc_bir_no_rtf))
    #     print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir_no_rtf)],
    #                                                    np.max(ave_acc_bir_no_rtf)))
    #     print("\n\nBI-R minus CONDITIONAL REPLAY")
    #     print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
    #     print("  {}".format(ave_acc_bir_no_con))
    #     print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir_no_con)],
    #                                                    np.max(ave_acc_bir_no_con)))
    #     print("\n\nBI-R minus INTERNAL REPLAY")
    #     print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
    #     print("  {}".format(ave_acc_bir_no_int))
    #     print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir_no_int)],
    #                                                    np.max(ave_acc_bir_no_int)))
    #     print("\n\nBI-R minus DISTILLATION")
    #     print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
    #     print("  {}".format(ave_acc_bir_no_dis))
    #     print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_bir_no_dis)],
    #                                                    np.max(ave_acc_bir_no_dis)))
    #     print("\n\nGR plus GATING BASED ON INTERNAL CONTEXT")
    #     print(" param-list (dg_prop): {}".format(dg_prop_list_onlybir))
    #     print("  {}".format(ave_acc_gr_plus_gat))
    #     print("--->  dg_prop = {}     --    {}".format(dg_prop_list_onlybir[np.argmax(ave_acc_gr_plus_gat)],
    #                                                    np.max(ave_acc_gr_plus_gat)))
    # print('\n')
    #
    #
    #
    # #-------------------------------------------------------------------------------------------------#
    #
    # #--------------------#
    # #----- PLOTTING -----#
    # #--------------------#
    #
    # # name for plot
    # plot_name = "hyperParams-{}{}-{}".format(args.experiment, args.tasks, args.scenario)
    # scheme = "incremental {} learning".format(args.scenario)
    # title = "{}  -  {}".format(args.experiment, scheme)
    # ylabel = "Average accuracy (after all tasks)"
    #
    # # calculate limits y-axes (to have equal for all graphs)
    # full_list = [item for sublist in ave_acc_per_lambda for item in sublist] + ave_acc_si + ave_acc_bir + \
    #             [item for sublist in ave_acc_bir_per_c for item in sublist]
    # if args.scenario=="task":
    #     full_list += ave_acc_xdg
    # if args.per_bir_comp:
    #     full_list += (ave_acc_bir_no_rtf + ave_acc_bir_no_con + ave_acc_bir_no_dis + ave_acc_bir_no_int
    #                   + ave_acc_gr_plus_gat)
    # miny = np.min(full_list)
    # maxy = np.max(full_list)
    # marginy = 0.1*(maxy-miny)
    #
    # # open pdf
    # pp = my_plt.open_pdf("{}/{}.pdf".format(args.p_dir, plot_name))
    # figure_list = []
    #
    #
    # ###---EWC + online EWC---###
    # # - select colors
    # colors = ["darkgreen"]
    # colors += get_cmap('Greens')(np.linspace(0.7, 0.3, len(gamma_list))).tolist()
    # # - make plot (line plot - only average)
    # figure = my_plt.plot_lines(ave_acc_per_lambda, x_axes=ext_lambda_list, ylabel=ylabel,
    #                            line_names=["EWC"] + ["Online EWC - gamma = {}".format(gamma) for gamma in gamma_list],
    #                            title=title, x_log=True, xlabel="EWC: lambda (log-scale)",
    #                            ylim=(miny-marginy, maxy+marginy),
    #                            with_dots=True, colors=colors, h_line=BASE, h_label="None")
    # figure_list.append(figure)
    #
    #
    # ###---SI---###
    # figure = my_plt.plot_lines([ave_acc_si], x_axes=ext_c_list, ylabel=ylabel, line_names=["SI"],
    #                         colors=["yellowgreen"], title=title, x_log=True, xlabel="SI: c (log-scale)", with_dots=True,
    #                         ylim=(miny-marginy, maxy+marginy), h_line=BASE, h_label="None")
    # figure_list.append(figure)
    #
    #
    # ###---XdG---###
    # if args.scenario=="task":
    #     figure = my_plt.plot_lines([ave_acc_xdg], x_axes=ext_xdg_list, ylabel=ylabel,
    #                             line_names=["XdG"], colors=["deepskyblue"], ylim=(miny-marginy, maxy+marginy),
    #                             title=title, x_log=False, xlabel="XdG: % of nodes gated",
    #                             with_dots=True, h_line=BASE, h_label="None")
    #     figure_list.append(figure)
    #
    #
    # ###---Brain-Inspired Replay---###
    # figure = my_plt.plot_lines([ave_acc_bir], x_axes=dg_prop_list_onlybir, ylabel=ylabel,
    #                            line_names=["Brain-Inspired Replay (BI-R)"],
    #                            colors=["purple"], title=title, x_log=False, xlabel="Context gates: % of nodes gated",
    #                            with_dots=True, ylim=(miny-marginy, maxy+marginy), h_lines=[BASE], h_labels=["None"],
    #                            h_colors=["grey"])
    # figure_list.append(figure)
    #
    #
    # ###---Brain-Inspired Replay + SI---###
    # # - select colors
    # colors = get_cmap('Blues_r')(np.linspace(0.6, 0., len(dg_prop_list))).tolist()
    # # - make plot (line plot - only average)
    # figure = my_plt.plot_lines(ave_acc_bir_per_c, x_axes=ext_c_list, ylabel=ylabel,
    #                            line_names=["BI-R, gate-prop = {}".format(dg_prop) for dg_prop in dg_prop_list],
    #                            title=title, x_log=True, xlabel="BI-R + SI: c (log-scale)",
    #                            ylim=(miny-marginy, maxy+marginy),
    #                            with_dots=True, colors=colors, h_line=BASE, h_label="None")
    # figure_list.append(figure)
    #
    #
    # ###---BI-R per component---###
    # if args.per_bir_comp:
    #     figure = my_plt.plot_lines([ave_acc_bir_no_rtf, ave_acc_bir_no_con, ave_acc_bir_no_int, ave_acc_bir_no_dis,
    #                                 ave_acc_gr_plus_gat],
    #                                x_axes=dg_prop_list_onlybir, ylabel=ylabel,
    #                                line_names=["BI-R - rtf", "BI-R - con", "BI-R - int", "BI-R - dis", "GR + gat"],
    #                                colors=["maroon", "red", "darkgoldenrod", "green", "darkorange"], title=title,
    #                                x_log=False, xlabel="Context gates: % of nodes gated", with_dots=True,
    #                                ylim=(miny - marginy, maxy + marginy), h_lines=[BASE], h_labels=["None"],
    #                                h_colors=["grey"])
    #     figure_list.append(figure)
    #
    #
    # # add figures to pdf
    # for figure in figure_list:
    #     pp.savefig(figure)
    #
    # # close the pdf
    # pp.close()
    #
    # # Print name of generated plot on screen
    # print("\nGenerated plot: {}/{}.pdf\n".format(args.p_dir, plot_name))
