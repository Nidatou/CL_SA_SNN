import numpy as np
import os
from scipy.ndimage import zoom


if __name__ == "__main__":
    T = 8
    dataset = 'nmnist'
    data_root = f'./store/datasets/{dataset}/frames_number_{T}_split_by_number'
    train_root = os.path.join(data_root, "train")
    test_root = os.path.join(data_root, "test")

    for i in range(10):
        train_data_dir = os.path.join(train_root, str(i))
        train_list = os.listdir(train_data_dir)
        for file_name in train_list:
            file_pth = os.path.join(train_data_dir, file_name)
            data = np.load(file_pth, allow_pickle=True)
            new_dict = {}
            for key in data:
                if key == 'frames':
                    new_data = []
                    for t in range(data['frames'].shape[0]):
                        new_data.append(zoom(data['frames'][t, ...], (1, 0.5, 0.5), order=0))
                    new_data = np.stack(new_data, axis=0)
                    new_dict[key] = new_data
                else:
                    new_dict[key] = data[key]
            # np.savez_compressed(os.path.join(train_data_dir, "one.npz"), **new_dict)
            np.savez_compressed(file_pth, **new_dict)
            print(f"{file_pth} fixed")

        test_data_dir = os.path.join(test_root, str(i))
        test_list = os.listdir(test_data_dir)
        for file_name in test_list:
            file_pth = os.path.join(test_data_dir, file_name)
            data = np.load(file_pth, allow_pickle=True)
            new_dict = {}
            for key in data:
                if key == 'frames':
                    new_data = []
                    for t in range(data['frames'].shape[0]):
                        new_data.append(zoom(data['frames'][t, ...], (1, 0.5, 0.5), order=0))
                    new_data = np.stack(new_data, axis=0)
                    new_dict[key] = new_data
                else:
                    new_dict[key] = data[key]
            np.savez_compressed(file_pth, **new_dict)
            print(f"{file_pth} fixed")

