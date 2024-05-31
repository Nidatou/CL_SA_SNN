import copy
import os
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset
from data_utils.const import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS, data_root
from data_utils.custom import ReducedDataset, SubDataset, TransformedDataset, EmbeddingDataset, permutate_image_pixels
from spikingjelly.datasets.n_mnist import NMNIST


def get_dataset(name, type="train", download=True, capacity=None, permutation=None, dir=data_root,
                verbose=False, augment=False, normalize=False, target_transform=None, valid_prop=0.):
    data_name = 'mnist' if name in ('mnist28') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image_transformations to be applied
    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+'_norm']]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x, p=permutation:permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type == 'test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    if (type == 'train' or type == 'valid') and valid_prop > 0:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(valid_prop * dataset_size))
        if type == 'train':
            indices_to_use = indices[split:]
        elif type == 'valid':
            indices_to_use = indices[:split]
        dataset = ReducedDataset(dataset, indices_to_use)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


def get_spkdataset(name, type='train', download=True, capacity=None, permutation=None, dir=data_root,
                   verbose=False, target_transform=None, valid_prop=0., T=16):
    assert type in ['train', 'test', 'valid']
    dataset_class = AVAILABLE_DATASETS[name]
    transforms_list = []
    # transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    dataset_transform = transforms.Compose(transforms_list)

    dataset = dataset_class(
        f'{dir}/{name}', train=False if type == 'test' else True, data_type='frame', frames_number=T,
        split_by='number', transform=None, target_transform=target_transform
    )
    # dataset = Spike_transformDataset(dataset, transform=dataset_transform)

    if (type == 'train' or type == 'valid') and valid_prop > 0:
        dataset_size = len(dataset)
        indices = len(range(dataset_size))
        split = int(np.floor(valid_prop * dataset_size))
        if type == 'train':
            indices_to_use = indices[split:]
        elif type == 'valid':
            indices_to_use = indices[:split]
        else:
            raise NotImplementedError()
        dataset = ReducedDataset(dataset, indices_to_use)
    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


def get_embedding_dataset(name, type='train', capacity=None, verbose=False, dir=data_root, dataset_suffix='all_data_',
                          target_transform=None, valid_prop=0.):
    assert type in ['train', 'valid', 'test']
    dataset = EmbeddingDataset(data_root=dir, dataset_suffix=dataset_suffix,
                               transform=None, target_transform=target_transform,
                               train=False if type == 'test' else True)
    if (type == 'train' or type == 'valid') and valid_prop > 0:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floot(valid_prop * dataset_size))
        if type == 'train':
            indices_to_use = indices[split:]
        else:  # -> type == 'valid':
            indices_to_use = indices[:split]
        dataset = ReducedDataset(dataset, indices_to_use)

    if verbose:
        print(f" --> {name}: '{type}'-dataset consisting of {len(dataset)} samples")

    if capacity is not None and len(dataset) < capacity:
        dataset_copy = copy.deepcopy(dataset)
        dataset = ConcatDataset([dataset_copy for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset


def get_singletask_experiment(name, data_dir=data_root, normalize=False, augment=False, verbose=False, T=16):
    if name == "MNIST":
        data_type = 'mnist'
    elif name == "MNIST28":
        data_type = 'mnist28'
    elif name == "CIFAR10":
        data_type = "cifar10"
    elif name == "CIFAR100":
        data_type = "cifar100"
    elif name == "NMNIST":
        data_type = "nmnist"
    elif "Embeddings" in name:
        data_type = name.lower()
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # get the configuration dict and data-sets
    config = DATASET_CONFIGS[data_type]
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[data_type+'_denorm']
    if 'Embeddings' in name:
        trainset = get_embedding_dataset(name, type='train', dir=os.path.join(data_dir, name), verbose=verbose)
        testset = get_embedding_dataset(name, type='test', dir=os.path.join(data_dir, name), verbose=verbose)
    elif name in ['nmnist']:
        trainset = get_spkdataset(name, type='train', dir=os.path.join(data_dir, name), verbose=verbose, T=T)
        testset = get_spkdataset(name, type='test', dir=os.path.join(data_dir, name), verbose=verbose, T=T)
    else:
        trainset = get_dataset(data_type, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
        testset = get_dataset(data_type, type='test', dir=data_dir, verbose=verbose, normalize=normalize)

    return (trainset, testset), config


# the option 'normalize' and 'augment' only implemented for CIFAR-based experiments.
def get_multitask_experiment(name, scenario, tasks, data_dir=data_root, normalize=False, augment=False,
                             only_config=False, verbose=False, exception=False, only_test=False, T=16):
    if name == 'permMNISY':
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = 10
        if not only_config:
            # prepare for the dataset
            if not only_test:
                # prepare for the training dataset
                train_dataset = get_dataset('mnist', type='train', permutation=None, dir=data_dir,
                                            target_transform=None, verbose=verbose)
            test_dataset = get_dataset('mnist', type='test', permutation=None, dir=data_dir,
                                       target_transform=None, verbose=verbose)

            if exception:
                permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(tasks-1)]
            else:
                permutations = [np.random.permutation(config['size']**2) for _ in range(tasks)]

            train_datasets = []
            test_datasets = []
            for task_id, perm in enumerate(permutations):
                # permuted dataset keeps all the classes and need to shift the labels
                target_transform = transforms.Lambda(
                    lambda y, x=task_id: y + x * classes_per_task
                ) if scenario in ('task', 'class', 'all') else None
                if not only_test:
                    train_datasets.append(TransformedDataset(
                        train_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                        target_transform=target_transform
                    ))
                test_datasets.append(TransformedDataset(
                    test_dataset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                    target_transform=target_transform
                ))
    elif name == 'splitMNIST':
        if tasks > 10:
            raise ValueError("Experiment '{}' cannot have more tasks than the labels".format(name))
        # configurations
        config = DATASET_CONFIGS['mnist28']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare for the split-dataset
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            # print(permutation)
            # this seemed only to be used to shuffle the labels?
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))

            if not only_test:
                mnist_train = get_dataset('mnist28', type="train", dir=data_dir, target_transform=target_transform,
                                          verbose=verbose)
            mnist_test = get_dataset('mnist28', type="test", dir=data_dir, target_transform=target_transform,
                                     verbose=verbose)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            train_datasets = []
            test_datasets = []
            # when considering the Domain-IL task the labels should be transformed
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario == 'domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))
    elif name == 'splitNMNIST':
        if tasks > 10:
            raise ValueError("Experiment '{}' cannot have more tasks than the labels".format(name))
        # configurations
        config = DATASET_CONFIGS['nmnist']
        if T == 8:
            config = DATASET_CONFIGS['nmnist8']
        classes_per_task = int(np.floor(10 / tasks))
        if not only_config:
            # prepare for the split-dataset, shuffle the number for randomness
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))

            if not only_test:
                nmnist_train = get_spkdataset('nmnist', type='train', dir=data_dir, target_transform=target_transform,
                                              verbose=verbose, T=T)
            nmnist_test = get_spkdataset('nmnist', type='test', dir=data_dir, target_transform=target_transform,
                                         verbose=verbose, T=T)
            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            train_datasets = []
            test_datasets = []
            # when considering the Domain-IL task the labels should be transformed
            for labels in labels_per_task:
                target_transform = transforms.Lambda(
                    lambda y, x=labels[0]: y - x
                ) if scenario == 'domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(nmnist_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(nmnist_test, labels, target_transform=target_transform))
    elif name == "CIFAR10":
        if tasks > 10:
            raise ValueError(f"Experiment '{name}' cannot have more tasks than the labels")
        config = DATASET_CONFIGS['cifar10']
        classes_per_task = int(np.floor(10/tasks))
        if not only_config:
            permutation = np.random.permutation(list(range(10)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            if not only_test:
                cifar10_train = get_dataset('cifar10', type='train', dir=data_dir, normalize=normalize,
                                            augment=augment, target_transform=target_transform, verbose=verbose)
            cifar10_test = get_dataset('cifar10', type='test', dir=data_dir, normalize=normalize,
                                       target_transform=target_transform, verbose=verbose)
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split up
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if scenario == "domain" else None
                if not only_test:
                    train_datasets.append(SubDataset(cifar10_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cifar10_test, labels, target_transform=target_transform))
    elif name == 'CIFAR100':
        if tasks > 100:
            raise ValueError(f"Experiment '{name}' cannot have more tasks than the labels")
        config = DATASET_CONFIGS['cifar100']
        classes_per_task = int(np.floor(100/tasks))
        if not only_config:
            permutation = np.random.permutation(list(range(100)))
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            if not only_test:
                cifar100_train = get_dataset('cifar100', type="train", dir=data_dir, normalize=normalize,
                                             augment=augment, target_transform=target_transform, verbose=verbose)
            cifar100_test = get_dataset('cifar100', type='test', dir=data_dir, normalize=normalize,
                                        target_transform=target_transform, verbose=verbose)

            # generate labels-per-task
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split up
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if scenario=='domain' else None
                if not only_test:
                    train_datasets.append(SubDataset(cifar100_train, labels, target_transform=target_transform))
                test_datasets.append(SubDataset(cifar100_test, labels, target_transform=target_transform))
    elif "Embeddings" in name:
        config = DATASET_CONFIGS[name.lower()]
        if tasks > config['classes']:
            raise ValueError(f"Experiment '{name}' cannot have more tasks than the labels")
        total_classes = config['classes']
        classes_per_task = int(np.floor(total_classes / tasks))
        if not only_config:
            permutation = np.random.permutation(list(range(total_classes)))
            # print(permutation)
            target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
            if not only_test:
                cifar10_eb_train = get_embedding_dataset(name, type='train', dir=os.path.join(data_dir, name),
                                                         target_transform=target_transform, verbose=verbose)
            cifar10_eb_test = get_embedding_dataset(name, type='test', dir=os.path.join(data_dir, name),
                                                    target_transform=target_transform, verbose=verbose)
            labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
            ]
            # split up
            train_datasets = []
            test_datasets = []
            for labels in labels_per_task:
                target_transform = transforms.Lambda(lambda y, x=labels[0]: y - x) if scenario == "domain" else None
                if not only_test:
                    train_datasets.append(SubDataset(cifar10_eb_train, labels, target_transform=target_transform))
                    # print(f"{len(train_datasets)}:{len(train_datasets[-1])}")  # test for the length of each dataset
                test_datasets.append(SubDataset(cifar10_eb_test, labels, target_transform=target_transform))
    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    config['classes'] = classes_per_task if scenario == 'domain' else classes_per_task * tasks
    config["normalize"] = normalize if name=='CIFAR100' else False
    if config['normalize']:
        config['denomalize'] = AVAILABLE_TRANSFORMS['cifar100_denorm']

    return config if only_config else ((train_datasets, test_datasets), config, classes_per_task)
