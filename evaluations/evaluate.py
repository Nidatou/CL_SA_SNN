import numpy as np
from sklearn import manifold
import torch
from torch.utils.data import ConcatDataset
from utils import utils
from visual import plotting
from spikingjelly.activation_based import functional

from models.strange_attention import Strange_att


def initiate_progress_dict(n_tasks):
    progress_dict = {}
    progress_dict["all_tasks"] = [[] for _ in range(n_tasks)]
    progress_dict['average'] = []
    progress_dict['dead'] = []
    progress_dict['sparseness'] = []
    progress_dict['act_k'] = []
    progress_dict["x_iteration"] = []
    progress_dict["x_task"] = []

    progress_dict["all_classes_spk"] = []
    progress_dict["true_classes_spk"] = []
    progress_dict["process_weight"] = []

    return progress_dict


###################################################
# classifier evaluation ###########################
def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
             no_task_mask=False, task=None):
    device = model._device()
    cuda = model._is_on_cuda()
    model.eval()
    ################
    # 考虑一下XdG的问题
    ################
    if model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    for data, labels in data_loader:
        if test_size:
            if total_tested >= test_size:
                break
        data, labels = data.to(device), labels.to(device)
        labels = labels - allowed_classes[0] if allowed_classes is not None else labels
        with torch.no_grad():
            scores = model.classify(data, not_hidden=True)
            scores = scores if allowed_classes is None else scores[:, allowed_classes]
            _, predicted = torch.max(scores, 1)

        # -update statistics
        functional.reset_net(model)
        total_correct += (predicted == labels).sum().item()
        total_tested += len(labels)

    accuracy = total_correct / total_tested
    # if verbose:
    #     print('=> accuracy: {:.3f}'.format(accuracy))
    return accuracy


def validate_atten(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
                   no_task_mask=False, task=None):
    device = model._device()
    cuda = model._is_on_cuda()

    model.eval()
    ################
    # 考虑一下XdG的问题
    ################
    if model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    atten_record = None
    for data, labels in data_loader:
        if test_size:
            if total_tested >= test_size:
                break
        data, labels = data.to(device), labels.to(device)
        labels = labels - allowed_classes[0] if allowed_classes is not None else labels
        with torch.no_grad():
            scores = model.classify(data, record_atten=True)
            scores = scores if allowed_classes is None else scores[:, allowed_classes]
            _, predicted = torch.max(scores, 1)

        # -update statistics
        functional.reset_net(model)
        total_correct += (predicted == labels).sum().item()
        total_tested += len(labels)
        if atten_record is None:
            atten_record = model.fcPart.atten_record
        else:
            for idx in range(len(model.fcPart.atten_record)):
                atten_record[idx] += model.fcPart.atten_record[idx]

    accuracy = total_correct / total_tested

    atten_log = ""
    for idx in range(len(atten_record)):
        atten_record[idx] /= total_tested
        atten_log += "-layer{}: ave:{:.3f}, min:{:.3f}, max:{:.3f}".format(
            idx, np.average(atten_record[idx]), np.min(atten_record[idx]), np.max(atten_record[idx]))

    if verbose:
        print('=> accuracy: {:.3f}'.format(accuracy))
        print(f'=> {atten_log}')
    return accuracy


def check_sparsity(model, datasets, curr_task, scenario='class', classes_per_task=None,
                   batch_size=128, test_size=1024, verbose=True, no_task_mask=False):
    res_dict = {}
    total_tested = 0
    activate_spike = act_distrib = activate_neuron = 0
    for i in range(curr_task):
        if scenario == 'task':
            allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
        elif scenario == 'class':
            allowed_classes = list(range(classes_per_task * (curr_task)))
        else:
            allowed_classes = None

        device = model._device()
        cuda = model._is_on_cuda()

        model.eval()
        ################
        # 考虑一下XdG的问题
        ################
        if model.mask_dict is not None:
            if no_task_mask:
                model.reset_XdGmask()
            else:
                model.apply_XdGmask(task=i+1)
        val_dataset = ConcatDataset(datasets[:curr_task])
        data_loader = utils.get_data_loader(val_dataset, batch_size, cuda=cuda)
        one_tested = 0
        for data, labels in data_loader:
            if test_size:
                if one_tested >= test_size:
                    break
            data, labels = data.to(device), labels.to(device)
            labels = labels - allowed_classes[0] if allowed_classes is not None else labels
            with torch.no_grad():
                batch_spike = model.check_neuron_activation(data)

            functional.reset_net(model)
            activate_spike += torch.sum(batch_spike)
            activate_neuron += torch.sum(batch_spike > 0)
            act_distrib += torch.sum(batch_spike, dim=0)
            one_tested += len(labels)
        total_tested += one_tested
    sparseness = activate_spike / total_tested
    k_neuron = activate_neuron / total_tested
    distrib = act_distrib / total_tested
    dead = (distrib <= 0).sum() / distrib.shape[0]
    if verbose:
        print(f"=>sparseness: {sparseness}, act_k:{k_neuron}, dead_prop:{dead}")
    res_dict['sparseness'] = sparseness.cpu().numpy()
    res_dict['act_k'] = k_neuron.cpu().numpy()
    res_dict['dead'] = dead.cpu().numpy()
    return res_dict


def check_specific_sparsity(model, datasets, curr_task, scenario='class', classes_per_task=None,
                            batch_size=128, test_size=None, no_task_mask=False, mid_size=None):
    assert mid_size is not None
    curr_class = classes_per_task * curr_task
    true_spk_res = np.zeros((curr_class, mid_size))
    total_spk_res = np.zeros((curr_class, mid_size))
    for i in range(curr_task):
        if scenario == 'task':
            allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
        elif scenario == 'class':
            allowed_classes = list(range(classes_per_task * (curr_task)))
        else:
            allowed_classes = None

        device = model._device()
        cuda = model._is_on_cuda()

        model.eval()
        ################
        # 考虑一下XdG的问题
        ################
        # if model.mask_dict is not None:
        #     if no_task_mask:
        #         model.reset_XdGmask()
        #     else:
        #         model.apply_XdGmask(task=i + 1)
        val_dataset = ConcatDataset(datasets[:curr_task])
        data_loader = utils.get_data_loader(val_dataset, batch_size, cuda=cuda)
        one_tested = 0
        for data, labels in data_loader:
            if test_size:
                if one_tested > test_size:
                    break
            data, labels = data.to(device), labels.to(device)
            batch_size = data.shape[0] if len(data.shape) == 2 or len(data.shape) == 4 else data.shape[1]
            labels = labels - allowed_classes[0] if allowed_classes is not None else labels
            with torch.no_grad():
                res = model.classify(data, not_hidden=True, return_spk=True)
                assert isinstance(res, tuple)
                scores, spk = res[0], res[1].cpu().numpy()

                scores = scores if allowed_classes is None else scores[:, allowed_classes]
                _, predicted = torch.max(scores, 1)
                for ind in range(batch_size):
                    total_spk_res[labels[ind]] += spk[ind]
                    if labels[ind] == predicted[ind]:
                        true_spk_res[labels[ind]] += spk[ind]

            functional.reset_net(model)

    return true_spk_res, total_spk_res


def test_accuracy(model, datasets, current_task, iteration, classes_per_task=None, scenario="none",
                  progress_dict=None, test_size=None, visdom=None, verbose=False, no_task_mask=False,
                  all_task=False):
    """Evaluate accuracy on all tasks so far

    :param classes_per_task: number of active classes per task
    :param scenario: the Cl scenario
    :param progress_dict: None or <dict> of all measures to keep track of
    :param test_size: the size of test samples
    :param visdom: None or <dict> with name of "graph" and "env", the remote visualization package
    :param verbose: whether to print process information
    :param no_task_mask: used for XdG
    """
    n_tasks = len(datasets)
    accs = []
    current_task = n_tasks if all_task else current_task
    for i in range(n_tasks):
        if i + 1 <= current_task:
            if scenario == 'task':
                allowed_classes = list(range(classes_per_task*i, classes_per_task*(i+1)))
            elif scenario == 'class':
                allowed_classes = list(range(classes_per_task*(current_task)))
            else:
                allowed_classes = None
            # if isinstance(model, Strange_att):
            #     accuracy = validate_atten(model, datasets[i], test_size=test_size, verbose=verbose,
            #                               allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i+1)
            # else:
            accuracy = validate(model, datasets[i], test_size=test_size, verbose=verbose,
                                allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i+1)
            accs.append(accuracy)

        else:
            accs.append(0)
    average_accs = sum(
        [accs[task_id] for task_id in range(current_task)]
    ) / current_task

    if verbose:
        print(' => ave accuracy: {:.3f}'.format(average_accs))

    names = [f'task {i + 1}' for i in range(n_tasks)]
    if visdom is not None:
        # when online visualization, it should pass the accuracy to it
        pass

    # verify the sparseness for teh model
    if hasattr(model, "check_neuron_activation"):
        res_dict = check_sparsity(model, datasets, curr_task=current_task, scenario=scenario, classes_per_task=classes_per_task,
                                  no_task_mask=no_task_mask, verbose=verbose)
    else:
        res_dict = None

    if progress_dict is not None:
        for task_id, _ in enumerate(names):
            progress_dict["all_tasks"][task_id].append(accs[task_id])
        progress_dict["average"].append(average_accs)
        progress_dict['x_iteration'].append(iteration)
        progress_dict['x_task'].append(current_task)
        if res_dict is not None:
            progress_dict['dead'].append(res_dict['dead'])
            progress_dict['act_k'].append(res_dict['act_k'])
            progress_dict['sparseness'].append(res_dict['sparseness'])
        else:
            progress_dict['dead'].append(0)
            progress_dict['act_k'].append(0)
            progress_dict['sparseness'].append(0)
    return progress_dict


def check_neuron_activation(
    model, datasets, current_task, classes_per_task=None, scenario="none",
    progress_dict=None, test_size=None, no_task_mask=False, all_task=False, mid_size=None
):
    n_tasks = len(datasets)
    n_classes = classes_per_task * n_tasks
    assert progress_dict is not None
    progress_dict["process_weight"].append(model.classifier_weight)
    current_task = n_tasks if all_task else current_task
    true_spk_res, total_spk_res = check_specific_sparsity(
        model, datasets, current_task, scenario=scenario, classes_per_task=classes_per_task, test_size=test_size,
        no_task_mask=no_task_mask, mid_size=mid_size
    )
    progress_dict["true_classes_spk"].append(true_spk_res)
    progress_dict["all_classes_spk"].append(total_spk_res)
    return progress_dict


###################################################
# generation evaluation ###########################
def show_samples(model, config, pdf=None, visdom=None, size=32, sample_mode=None, title="Generated samples",
                 allowed_classes=None, allowed_domains=None):
    '''Plot samples from a generative model in [pdf] and/or in [visdom].'''

    # Set model to evaluation-mode
    model.eval()

    # Generate samples from the model
    sample = model.sample(size, sample_mode=sample_mode, allowed_classes=allowed_classes,
                          allowed_domains=allowed_domains, only_x=True)
    # -correctly arrange pixel-values and move to cpu (if needed)
    image_tensor = sample.view(-1, config['channels'], config['size'], config['size']).cpu()
    # -denormalize images if needed
    if config['normalize']:
        image_tensor = config['denormalize'](image_tensor).clamp(min=0, max=1)

    # Plot generated images in [pdf] and/or [visdom]
    # -number of rows
    nrow = int(np.ceil(np.sqrt(size)))
    # -make plots
    if pdf is not None:
        plotting.plot_images_from_tensor(image_tensor, pdf, title=title, nrow=nrow)
    # if visdom is not None:
